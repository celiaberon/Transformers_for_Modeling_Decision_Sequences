import itertools
import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, '../../')))
sys.path.append(os.path.abspath(os.path.join(__file__, '../../transformer/')))

import numpy as np
import pandas as pd
import torch

import utils.file_management as fm
from synthetic_data_generation.agent import RFLR_mouse
from transformer.transformer import GPT, GPTConfig

logger = None

def get_data_filenames(run, suffix='tr'):
    behavior_filename = fm.get_experiment_file("behavior_run_{}.txt", run, suffix, subdir='seqs')
    high_port_filename = fm.get_experiment_file("high_port_run_{}.txt", run, suffix, subdir='seqs')
    session_filename = fm.get_experiment_file("session_transitions_run_{}.txt", run, suffix, subdir='seqs')
    assert fm.check_files_exist(behavior_filename, high_port_filename, session_filename)
    return behavior_filename, high_port_filename, session_filename


def parse_simulated_data(behavior_filename, high_port_filename, session_filename,
                         clip_short_blocks=False):
    """Parse simulated data from behavior, high port, and session files.

    Args:
        behavior_filename (str): Path to behavior file.
        high_port_filename (str): Path to high port file.
        session_filename (str): Path to session file.
        clip_short_blocks (bool): Whether to clip short blocks.

    Returns:
        pd.DataFrame: Parsed data with added sequence columns.

    Notes: Typically for simulated data used as input for training model.
    """

    run_and_suffix = behavior_filename.split('_')[-1].split('.')[0]
    suffix = 'tr' if run_and_suffix.endswith('tr') else 'v'
    run = int(run_and_suffix[:-len(suffix)])

    behavior_data = fm.read_sequence(behavior_filename)
    high_port_data = fm.read_sequence(high_port_filename)
    assert len(behavior_data) == len(high_port_data), (
        "Error: Behavior data and high port data have different lengths.")

    token = list(behavior_data)
    choices = [int(c in ['R', 'r']) for c in behavior_data]
    choices_str = [c.upper() for c in behavior_data]
    rewards = [int(c.isupper()) for c in behavior_data]
    trials = np.arange(len(choices)).astype('int')
    high_ports = [int(hp) for hp in high_port_data]
    selected_high = [c == hp for c, hp in zip(choices, high_ports)]
    switch = np.abs(np.diff(choices, prepend=np.nan))
    transitions = np.abs(np.diff(high_ports, prepend=np.nan))

    events = pd.DataFrame(data={
        'trial_number': trials,
        'k0': token,
        'choice': choices,
        'choice_str': choices_str,
        'reward': rewards,
        'selected_high': selected_high,
        'switch': switch,
        'transition': transitions,
        'high_port': high_ports,
    })

    session_df = pd.read_csv(session_filename, names=['trial_number', 'domain'])
    # Forward fill sessions and domains for all trials
    full_sessions = pd.merge_asof(
        events[['trial_number']],
        session_df.assign(session=np.arange(len(session_df))),
        on='trial_number',
        direction='backward'
    )
    events['domain'] = full_sessions['domain']
    events['session'] = full_sessions['session']

    # First trial in a session cannot be a switch or transition.
    events.loc[events['trial_number'].isin(session_df['trial_number']), 'switch'] = np.nan
    events.loc[events['trial_number'].isin(session_df['trial_number']), 'transition'] = np.nan

    events = get_block_positions(events)
    events = get_previous_block_length(events)
    events = add_sequence_columns(events, seq_length=2)
    events = add_sequence_columns(events, seq_length=3)
    events = parse_passive_estimator(run, events, suffix=suffix)

    if clip_short_blocks:
        raise NotImplementedError

    return events


def parse_passive_estimator(run, events, suffix='tr'):
    """Parse passive estimator data from events."""
    # Initialize columns
    events['phi'] = None
    events['logodds'] = None
    events['pright'] = None

    # Get unique domains and their parameters
    domains = events.domain.unique()
    params = {}
    for domain in domains:
        params[domain] = fm.get_domain_params(run=run, domain_id=domain, suffix=suffix)[1]

    for session in events.session.unique():
        session_mask = events.session == session
        session_events = events[session_mask]
        domain_id = session_events.domain.unique().item()
        domain_params = params[domain_id]
        agent = RFLR_mouse(**domain_params)
        
        # Get full session data
        choices = session_events.choice.values
        rewards = session_events.reward.values
        
        # Process full session at once
        phi_values, logodds_values, pright_values = agent.passive_estimator(
            choices, rewards
        )
        
        events.loc[session_mask, 'phi'] = phi_values
        events.loc[session_mask, 'logodds'] = logodds_values
        events.loc[session_mask, 'pright'] = pright_values

    return events


def parse_passive_estimator_qlearning(run, events, suffix='tr'):
    """Parse Q-learning passive estimator data from events."""
    # Initialize columns
    events['rpe'] = None
    events['q_left'] = None
    events['q_right'] = None
    events['q_logodds'] = None
    events['q_pright'] = None

    # Get unique domains and their parameters
    domains = events.domain.unique()
    params = {}
    for domain in domains:
        params[domain] = fm.get_domain_params(run=run, domain_id=domain, suffix=suffix)[1]

    for session in events.session.unique():
        session_mask = events.session == session
        session_events = events[session_mask]
        domain_id = session_events.domain.unique().item()
        domain_params = params[domain_id]
        agent = RFLR_mouse(**domain_params)
        
        choices = session_events.choice.values
        rewards = session_events.reward.values
        
        rpes, qs, logodds, pright = agent.passive_estimator_qlearning(
            choices, rewards
        )
        
        # Assign all values at once using the mask
        events.loc[session_mask, 'rpe'] = rpes
        events.loc[session_mask, 'q_left'] = qs[:, 0]
        events.loc[session_mask, 'q_right'] = qs[:, 1]
        events.loc[session_mask, 'q_logodds'] = logodds
        events.loc[session_mask, 'q_pright'] = pright

    return events


def get_previous_block_length(events):
    # ID of short blocks, and blocks immediately following short blocks (not back to baseline).
    # short_blocks = events.query('block_length < 20')['block_id'].unique()
    # post_short_blocks = short_blocks + 1

    block_info = events[['session', 'block_id', 'block_length']].copy().drop_duplicates()
    block_info['prev_block_length'] = block_info.groupby('session')['block_length'].shift(1)

    events = events.merge(block_info[['session', 'block_id', 'prev_block_length']],
                          on=['session', 'block_id'], how='left')
    # For reference:
    # sns.lineplot(data=events.query('block_position.between(0, 20) & ~block_id.isin(@post_short_blocks)'), x='block_position', y='switch', ax=ax)
    # sns.lineplot(data=events.query('rev_block_position.between(-10, 1) & block_length > 20'), x='rev_block_position', y='switch', ax=ax)

    return events


def get_block_positions(events):

    """Calculate block-related metrics, treating each session independently."""
    # Group by session and calculate block information
    session_col = events['session'].copy()
    events_with_blocks = (events
                          .groupby('session', group_keys=False)
                          .apply(lambda session_events: _get_session_block_positions(session_events), include_groups=False))
    events_with_blocks['session'] = session_col

    return events_with_blocks


def _get_session_block_positions(session_events):
    """Helper function to calculate block positions within a single session."""
    # Get transition points within this session

    first_trial = session_events.iloc[0]['trial_number']
    transition_points = session_events.query('transition == 1')['trial_number'].values - first_trial

    # Calculate block lengths
    if len(transition_points) == 0:
        # Single block session
        block_lengths = [len(session_events)]
    else:
        # First block
        block_lengths = [transition_points[0]]
        # Middle blocks
        block_lengths.extend(np.diff(transition_points).astype('int'))
        # Last block
        block_lengths.extend([len(session_events) - transition_points[-1]])
    
    # Calculate length of each block as num trial from previous transition.
    # Prepend first block, and last block is distance to end of sequence.
    # block_lengths = [events.query('transition == 1')['trial_number'].values[0]]
    # block_lengths.extend(events.query('transition == 1')['trial_number'].diff().values[1:].astype('int'))
    # block_lengths.extend([len(events) - events.query('transition == 1')['trial_number'].values[-1]])

    # Store block lengths at transitions and fill backwards (so each trial can
    # reference ultimate block length).
    session_events.loc[session_events.index[0], 'block_length'] = block_lengths[0]
    if len(transition_points) > 0:
        session_events.loc[session_events['transition'] == 1, 'block_length'] = block_lengths[1:]
    session_events['block_length'] = session_events['block_length'].ffill()

    # Counter for index position within each block. Forward and reverse
    # (negative, from end of block backwards).
    block_positions = list(itertools.chain(*[np.arange(i) for i in block_lengths]))
    session_events['block_position'] = block_positions
    session_events['rev_block_position'] = session_events['block_position'] - session_events['block_length']

    # Unique ID for each block.
    if len(transition_points) > 0:
        session_events.loc[session_events.index[0], 'block_id'] = 0
        session_events.loc[session_events['transition'] == 1, 'block_id'] = np.arange(1, len(block_lengths))
        session_events['block_id'] = session_events['block_id'].ffill()

    else:
        session_events['block_id'] = 0
    return session_events


def map_sequence_to_pattern(seq):
    """Maps a sequence of actions to a pattern string (encoding).

    Takes a sequence of actions (dictionaries containing choice and reward
    info) and converts it into a pattern string using the following rules:
    - First action: 'A' if rewarded, 'a' if unrewarded
    - Subsequent actions relative to first choice:
        - Same side as first: 'A' if rewarded, 'a' if unrewarded
        - Different side: 'B' if rewarded, 'b' if unrewarded

    Args:
        seq: Dictionary or dataframe, containing:
            - choice_str: String indicating choice ('L' or 'R')
            - rewarded: Boolean indicating if choice was rewarded

    Returns:
        str: Pattern string encoding the sequence (e.g. 'aAb')
    """
    action1, *actionN = seq

    # First action: 'A' if rewarded, 'a' if unrewarded
    first_letter = 'A' if action1['rewarded'] else 'a'
    first_choice = action1['choice_str']
    pattern = first_letter
    # Subsequent actions

    for i_action in actionN:
        same_side = i_action['choice_str'] == first_choice
        if same_side:
            next_letter = 'A' if i_action['rewarded'] else 'a'
        else:
            next_letter = 'B' if i_action['rewarded'] else 'b'
        pattern += next_letter

    return pattern


def map_rl_to_pattern(seq):
    """Maps a sequence of actions to a pattern string (encoding).

    Takes a sequence of actions already encoded as ['R', 'r', 'L', 'L'] and
    converts it into a pattern string using the following rules:
    - First action: 'A' if rewarded, 'a' if unrewarded
    - Subsequent actions relative to first choice:
        - Same side as first: 'A' if rewarded, 'a' if unrewarded
        - Different side: 'B' if rewarded, 'b' if unrewarded

    Args:
        seq: List, tuple, or string, containing at least two characters.
        Expects encoded actions/outcomes as ['R', 'r', 'L', 'L'].

    Returns:
        str: Pattern string encoding the sequence (e.g. 'aAb')
    """
    action1, *actionN = seq
    first_letter = 'A' if action1.isupper() else 'a'
    first_choice = action1.upper()
    pattern = first_letter
    for i_action in actionN:
        same_side = i_action.upper() == first_choice
        if same_side:
            next_letter = 'A' if i_action.isupper() else 'a'
        else:
            next_letter = 'B' if i_action.isupper() else 'b'
        pattern += next_letter
    return pattern


def add_sequence_columns(events, seq_length):
    """Add sequence columns (history up to but NOT INCLUDING current trial) to
    events DataFrame.

    For a given sequence length N, adds two columns to track trial histories:
    - seqN_RL: right/left encoded sequence of choices/rewards for previous N
               trials (e.g. 'RrL')
    - seqN: Pattern-encoded sequence using A/a/B/b notation (e.g. 'aAb')

    Args:
        events (pd.DataFrame): DataFrame containing trial data with column 'k0' 
            encoding choices/rewards
        seq_length (int): Number of previous trials to include in sequence

    Returns:
        pd.DataFrame: Original DataFrame with added sequence columns. seqN
        is pattern of N previous trials UP TO but NOT INCLUDING the current
        trial.
    """

    events[[f'seq{seq_length}_RL', f'seq{seq_length}']] = None

    # Group by session and calculate sequences
    def get_session_sequences(session_events):
        if len(session_events) >= seq_length:
            start_idx = session_events.index[0] + seq_length
            session_events.loc[start_idx:, f'seq{seq_length}_RL'] = [
                ''.join(session_events['k0'].values[i-seq_length:i])
                for i in range(seq_length, len(session_events))
            ]
            session_events.loc[start_idx:, f'seq{seq_length}'] = (
                session_events.loc[start_idx:, f'seq{seq_length}_RL']
                .apply(map_rl_to_pattern)
            )
        return session_events

    events = events.groupby('session', group_keys=False).apply(get_session_sequences)

    return events


def align_predictions_with_gt(events, predictions, indices=None):
    """Align predictions from trained model (e.g. transformer) with ground
    truth events assuming token-wise predictions.

    Args:
        events (pd.DataFrame): Ground truth events.
        predictions (list or str): Predictions.

    Returns:
        pd.DataFrame: Aligned events with predictions.
    """

    if indices is None:
        indices = events.index

    assert len(indices) == len(predictions), (
        'indices and predictions must have the same length')

    events_ = events.copy()
    events_['pred_k0'] = None
    events_.loc[indices, 'pred_k0'] = [p for p in predictions]
    events_['pred_choice'] = pd.Series(dtype='Int64')
    events_.loc[indices, 'pred_choice'] = [c in ['R', 'r'] for c in predictions]
    events_.loc[indices, 'pred_choice_str'] = events_.loc[indices, 'pred_k0'].str.upper()
    events_['pred_reward'] = pd.Series(dtype='Int64')
    events_.loc[indices, 'pred_reward'] = [c.isupper() for c in predictions]

    events_['pred_selected_high'] = (events_['pred_choice'] == events_['high_port']).astype('Int64')
    events_['prev_choice'] = events_['seq2_RL'].apply(lambda x: x[-1].upper()
                                                      if not pd.isna(x) else None)
    events_['pred_switch'] = ((events_['pred_choice_str'] != events_['prev_choice'])
                              .where(events_['prev_choice'].notna()
                                     & events_['pred_choice_str'].notna(), np.nan)
                              .astype('Int64'))
    events_['pred_correct_k0'] = events_['pred_k0'] == events_['k0']
    events_['pred_correct_choice'] = events_['pred_choice'] == events_['choice']

    return events_


def load_trained_model(run, model_name, device, **kwargs):

    # Get model info from metadata
    model_info = fm.parse_model_info(run, model_name=model_name)
    if model_name is None:
        model_name = model_info['model_name']
    else:
        assert (model_info['model_name'] == model_name) or (model_info['model_name'] == model_name.split('_cp')[0]), (
            'did not recover correct model')

    # Configure model using metadata
    config = GPTConfig(
        block_size=model_info['config'].get('Block size', 12),
        vocab_size=model_info['config'].get('Vocab size', 4),
        n_layer=model_info['config'].get('Number of layers', 1),
        n_head=model_info['config'].get('Number of heads', 1),
        n_embd=model_info['config'].get('Embedding size', 64)
    )
    # Load the trained model
    model = GPT(config)
    model_path = fm.get_experiment_file(f'{model_name}.pth', run, subdir='models')

    try:
        model.load_state_dict(torch.load(model_path, map_location=device, **kwargs))
    except Exception:
        checkpoint = torch.load(model_path, map_location=device,  **kwargs)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, model_info, config


def load_predictions(run, model_name, suffix='v'):

    files = get_data_filenames(run, suffix=suffix)
    gt_events = parse_simulated_data(*files)

    predictions_filename = fm.get_experiment_file(f"pred_{model_name}.txt", run, subdir='seqs')
    indices_filename = fm.get_experiment_file(f"pred_indices_{model_name}.txt", run, subdir='seqs')
    assert fm.check_files_exist(predictions_filename)

    predictions = fm.read_sequence(predictions_filename)

    indices = None
    try:
        with open(indices_filename, 'r') as f:
            indices = [int(line.strip()) for line in f]
    except FileNotFoundError:
        pass

    # Align predictions with ground truth
    aligned_data = align_predictions_with_gt(gt_events, predictions, indices)

    if logger is not None:
        logger.info(f"Analyzing data from:\n {f}\n" for f in files)
        logger.info(f"Loading model predictions from: {predictions_filename}")

        if indices is not None:
            logger.info(f"Using indices file for alignment: {indices_filename}")
        else:
            logger.info("No indices file found, using sequential alignment")

        logger.info(f"Number of events: {len(gt_events)}")
        logger.info(f"Number of predictions: {len(predictions)}")

        logger.info(f"Aligned {len(aligned_data)} events with predictions")

    return aligned_data
