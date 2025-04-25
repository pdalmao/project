import can
import time
import os
import datetime
import numpy as np
import operator as op
import argparse
from collections import deque, defaultdict
import joblib  # For loading trained models

# --- Constants ---
FREQ_WINDOW_SIZE = 10      # Messages window for frequency calculation
MIN_MSGS_FOR_FREQ = 2      # Min messages needed to calculate frequency
RULE_RELOAD_INTERVAL = 60  # Seconds - How often to reload rules
DEFAULT_MODEL_DIR = "ml_models" # Default ML model directory

def calculate_frequency(timestamps):
    """Calculates the median frequency based on a deque of timestamps."""
    if len(timestamps) < MIN_MSGS_FOR_FREQ:
        return None
    # Calculate positive intervals between consecutive timestamps
    intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1) if timestamps[i+1] - timestamps[i] > 0]
    if not intervals:
        return None # Avoid errors if no valid intervals
    median_interval = np.median(intervals)
    # Return frequency (1/interval), handle potential zero interval
    return float('inf') if median_interval <= 0 else 1.0 / median_interval

def parse_rules(filepath):
    """Parses rules from a text file (e.g., can_rules.txt)."""
    invalid = 0
    rules = defaultdict(list) # Stores rules keyed by CAN ID
    num_rules = 0

    if not os.path.exists(filepath):
        print(f"[ERROR] Rule file not found: {filepath}")
        return None, 0

    try:
        # First pass to count potential rules (non-comment/empty lines)
        with open(filepath, 'r') as f:
            num_rules = sum(1 for line in f if line.strip() and not line.strip().startswith('#'))

        # Second pass to parse rules
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'): # Skip empty lines and comments
                    continue

                parts = line.split()
                try:
                    rule_type = parts[0].upper()
                    can_ids = [] # IDs this rule applies to

                    # Parse CAN ID or ID range (common to most rules)
                    if rule_type in ['ALERT', 'FREQ_ALERT', 'LENGTH_ALERT']:
                        can_id_str = parts[1]
                        if '-' in can_id_str: # Handle ID range (e.g., 0x100-0x1FF)
                            start_id_str, end_id_str = can_id_str.split('-')
                            start_id = int(start_id_str, 0) # Base 0 auto-detects hex/dec
                            end_id = int(end_id_str, 0)
                            can_ids = range(start_id, end_id + 1)
                        else: # Single ID
                            can_ids = [int(can_id_str, 0)]

                    # Rule Specific Parsing
                    if rule_type == 'ALERT': # Data/Bitmask checks
                        byte_spec_str, op1, *rest = parts[2:]
                        byte_spec = tuple(int(b) for b in byte_spec_str.split('-')) if '-' in byte_spec_str else int(byte_spec_str)

                        if op1.startswith('&'): # Bitmask rule (e.g., &0xF0 == 0x10)
                            mask = int(op1[1:], 0)
                            operator = rest[0].lower()
                            value = int(rest[1], 0)
                            message = ' '.join(rest[2:])
                            rule_detail = {'type': 'bitmask', 'byte_index': byte_spec if isinstance(byte_spec, int) else byte_spec[0],
                                           'mask': mask, 'operator': operator, 'value': value, 'message': message}
                        else: # Standard data rule (e.g., 3 > 100)
                            operator = op1.lower()
                            value_str = rest[0]
                            message = ' '.join(rest[1:])
                            if operator in ['in', 'notin']: # Handle 'in'/'notin' with comma-separated values
                                value = set(int(v, 0) for v in value_str.split(','))
                            elif operator in ['==', '!=', '>', '<', '>=', '<=']:
                                value = int(value_str, 0)
                            else: raise ValueError(f"Unsupported operator '{operator}'")
                            rule_detail = {'type': 'data', 'byte_spec': byte_spec, 'operator': operator,
                                           'value': value, 'message': message}

                        for can_id in can_ids: rules[can_id].append(rule_detail)

                    elif rule_type == 'FREQ_ALERT': # Frequency checks
                        min_freq, max_freq = float(parts[2]), float(parts[3])
                        message = ' '.join(parts[4:])
                        for can_id in can_ids:
                            rules[can_id].append({'type': 'frequency', 'min_freq': min_freq, 'max_freq': max_freq, 'message': message})

                    elif rule_type == 'LENGTH_ALERT': # DLC checks
                        max_length = int(parts[2])
                        message = ' '.join(parts[3:])
                        for can_id in can_ids:
                            rules[can_id].append({'type': 'length', 'max_length': max_length, 'message': message})
                    else:
                        raise ValueError(f"Unknown rule type '{rule_type}'")

                except (IndexError, ValueError, Exception) as e:
                    print(f"[WARN] Skipping invalid rule on line {line_num}: \"{line}\" - {e}")
                    invalid += 1
                    continue # Skip to next line

        valid_rule_count = num_rules - invalid
        print(f"[INFO] Loaded {valid_rule_count} valid rules out of {num_rules} from {filepath}")
        return rules, valid_rule_count

    except Exception as e:
        print(f"[ERROR] Failed to read rule file {filepath}: {e}")
        return None, 0

# Operator maps for rule evaluation
op_map_data = {'==': op.eq, '!=': op.ne, '>': op.gt, '<': op.lt, '>=': op.ge, '<=': op.le,
               'in': lambda a, b: a in b, 'notin': lambda a, b: a not in b}
op_map_mask = {'==': op.eq, '!=': op.ne, '>': op.gt, '<': op.lt, '>=': op.ge, '<=': op.le}

def evaluate_data_rule(data, rule):
    """Evaluates data rules (value comparisons). Handles single bytes and ranges."""
    index = rule['byte_spec']
    try:
        if isinstance(index, tuple): # Byte range
            start, end = index
            if end >= len(data): return False # Check bounds
            value_to_check = int.from_bytes(data[start:end+1], byteorder='big')
        else: # Single byte
            if index >= len(data): return False # Check bounds
            value_to_check = data[index]
        return op_map_data[rule['operator']](value_to_check, rule['value'])
    except Exception as e: # Catch potential errors like invalid operator or index issues
        print(f"[ERROR] Error evaluating data rule: {e} - Rule: {rule}, Data: {data.hex()}")
        return False

def evaluate_bitmask_rule(data, rule):
    """Evaluates bitmask rules."""
    index = rule['byte_index']
    try:
        if index >= len(data): return False # Check bounds
        masked_value = data[index] & rule['mask']
        return op_map_mask[rule['operator']](masked_value, rule['value'])
    except Exception as e:
        print(f"[ERROR] Error evaluating bitmask rule: {e} - Rule: {rule}, Data: {data.hex()}")
        return False

def format_packet_for_alert(data, rule):
    """Formats packet data (hex) for alerts, highlighting triggering bytes/nibbles based on rule."""
    parts = []
    target_index = -1 # For bitmask byte
    mask = 0
    highlight_indices = set() # For data bytes/ranges

    # Determine highlighting based on the rule that triggered
    if rule['type'] == 'data':
        bs = rule['byte_spec']
        if isinstance(bs, tuple):
            start, end = bs
            if start < len(data) and end < len(data): highlight_indices.update(range(start, end + 1))
        elif isinstance(bs, int):
            if bs < len(data): highlight_indices.add(bs)
    elif rule['type'] == 'bitmask':
        if 'byte_index' in rule and rule['byte_index'] < len(data):
            target_index = rule['byte_index']
            mask = rule.get('mask', 0)

    # Format data with highlighting
    for i, byte in enumerate(data):
        hex_byte = f"{byte:02X}"
        if i == target_index: # Bitmask highlighting (check nibbles)
            high_nibble, low_nibble = hex_byte[0], hex_byte[1]
            high_mask, low_mask = mask & 0xF0, mask & 0x0F
            if high_mask and low_mask: parts.append(f"[{hex_byte}]") # Highlight whole byte
            elif high_mask: parts.append(f"[{high_nibble}]{low_nibble}") # Highlight high nibble
            elif low_mask: parts.append(f"{high_nibble}[{low_nibble}]") # Highlight low nibble
            else: parts.append(hex_byte) # No relevant mask bits
        elif i in highlight_indices: # Data rule highlighting
            parts.append(f"[{hex_byte}]") # Highlight whole byte
        else:
            parts.append(hex_byte)

    while len(parts) < 8: parts.append("  ") # Pad for alignment
    return ' '.join(parts)

def load_ml_models(model_dir):
    """Loads pre-trained ML models and scalers from a directory."""
    models = {}
    scalers = {}
    if not os.path.isdir(model_dir):
        print(f"[WARN] ML Model directory not found: {model_dir}. Anomaly detection disabled.")
        return models, scalers

    print(f"[INFO] Loading ML models from: {model_dir}")
    loaded_count = 0
    for filename in os.listdir(model_dir):
        try:
            file_path = os.path.join(model_dir, filename)
            # Expecting filenames like: model_0xCANID_type.joblib or scaler_0xCANID_type.joblib
            if filename.endswith(".joblib"):
                is_model = filename.startswith("model_")
                is_scaler = filename.startswith("scaler_")
                if not (is_model or is_scaler): continue # Skip other files

                parts = filename[:-7].split('_') # Remove .joblib, split
                obj_type = parts[-1] # time or payload
                can_id_hex = parts[-2]
                can_id = int(can_id_hex, 0)
                loaded_obj = joblib.load(file_path)

                if is_model:
                    if can_id not in models: models[can_id] = {}
                    models[can_id][obj_type] = loaded_obj
                    loaded_count +=1
                elif is_scaler:
                    if can_id not in scalers: scalers[can_id] = {}
                    scalers[can_id][obj_type] = loaded_obj
        except Exception as e:
            print(f"[WARN] Failed to load ML model/scaler {filename}: {e}")
    print(f"[INFO] Loaded {loaded_count} ML models.")
    return models, scalers

def run_hybrid_ids(rule_file, model_dir, channel, log_file_path, print_to_console):
    """Runs the hybrid IDS, monitoring CAN traffic using rules and ML models."""
    # Initialization
    rules, rule_count = parse_rules(rule_file)
    if rules is None: # Handle rule loading failure
       if not os.path.exists(model_dir) or not os.listdir(model_dir): # Exit if no models either
           print("[ERROR] Failed to load rules and no ML models found. Exiting.")
           return
       else: # Warn if continuing with only ML
           print("[WARN] Failed to load rules. Running with only anomaly detection.")
           rules = defaultdict(list) # Ensure rules is a dict

    ml_models, ml_scalers = load_ml_models(model_dir)
    if not rules and not ml_models: # Check again after model loading
        print("[ERROR] No rules or ML models loaded. Nothing to detect. Exiting.")
        return

    timestamps = defaultdict(lambda: deque(maxlen=FREQ_WINDOW_SIZE)) # For freq calc
    last_msg_time = {} # For time delta calc
    last_reload_time = time.time()

    print(f"[INFO] Starting Hybrid IDS monitoring on {channel}...")
    try:
        with can.interface.Bus(channel=channel, interface='socketcan') as bus, \
             open(log_file_path, "a") as log_file:
            while True:
                # --- Rule Reload ---
                current_time = time.time()
                if current_time - last_reload_time > RULE_RELOAD_INTERVAL:
                    print("[INFO] Reloading rules...")
                    new_rules, _ = parse_rules(rule_file)
                    if new_rules is not None: rules = new_rules
                    else: print("[WARN] Failed to reload rules. Continuing with old rules.")
                    last_reload_time = current_time

                # --- Receive and Process Message ---
                msg = bus.recv(1.0) # 1s timeout allows checking reload interval
                if msg is None: continue

                now = msg.timestamp if msg.timestamp is not None else time.time()
                cid, data, data_len = msg.arbitration_id, msg.data, msg.dlc

                # Update state for frequency and time delta calculations
                timestamps[cid].append(now)
                time_delta = now - last_msg_time.get(cid, now)
                last_msg_time[cid] = now

                alert_triggered = False
                alert_reasons = [] # Collect all reasons for an alert

                # --- 1. Signature Check ---
                for rule in rules.get(cid, []):
                    rule_alert = False
                    violated_rule_msg = ""
                    try:
                        if rule['type'] == 'frequency':
                            freq = calculate_frequency(timestamps[cid])
                            if freq is not None and not (rule['min_freq'] <= freq <= rule['max_freq']):
                                rule_alert = True
                                freq_str = f"{freq:.2f}Hz" if freq != float('inf') else "inf Hz"
                                violated_rule_msg = rule['message'].replace("{freq}", freq_str)
                        elif rule['type'] == 'data' and evaluate_data_rule(data, rule):
                            rule_alert = True; violated_rule_msg = rule['message']
                        elif rule['type'] == 'bitmask' and evaluate_bitmask_rule(data, rule):
                            rule_alert = True; violated_rule_msg = rule['message']
                        elif rule['type'] == 'length' and data_len > rule['max_length']:
                            rule_alert = True; violated_rule_msg = rule['message'].replace("{length}", str(data_len))

                        if rule_alert:
                            alert_triggered = True
                            alert_reasons.append(f"RuleMatch: \"{violated_rule_msg}\"")
                            # Continue checking other rules for the same message
                    except Exception as e:
                        print(f"[ERROR] Exception during rule evaluation for ID 0x{cid:X}: {e}")

                # --- 2. Anomaly Check ---
                if cid in ml_models and cid in ml_scalers:
                    # Prepare features (pad payload, reshape for sklearn)
                    payload_features = list(data) + [0] * (8 - len(data))
                    time_feature = [[time_delta]]
                    payload_feature_vector = [payload_features]
                    anomaly_detected_here = False
                    anomaly_types_here = []

                    try:
                        # Time Anomaly
                        if 'time' in ml_models[cid] and 'time' in ml_scalers[cid]:
                            scaled_time = ml_scalers[cid]['time'].transform(time_feature)
                            if ml_models[cid]['time'].predict(scaled_time)[0] == -1:
                                anomaly_detected_here = True; anomaly_types_here.append("Time")
                        # Payload Anomaly
                        if 'payload' in ml_models[cid] and 'payload' in ml_scalers[cid]:
                             scaled_payload = ml_scalers[cid]['payload'].transform(payload_feature_vector)
                             if ml_models[cid]['payload'].predict(scaled_payload)[0] == -1:
                                 if not anomaly_detected_here: anomaly_detected_here = True # Set flag if not already set
                                 if "Payload" not in anomaly_types_here: anomaly_types_here.append("Payload")

                        if anomaly_detected_here:
                            alert_triggered = True
                            alert_reasons.append(f"Anomaly({','.join(anomaly_types_here)})")
                    except Exception as e:
                        print(f"[ERROR] Exception during anomaly prediction for ID 0x{cid:X}: {e}")

                # --- 3. Logging ---
                if alert_triggered:
                    t_str = datetime.datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    # Find first data/bitmask rule match for highlighting context, default if none
                    triggering_rule = next((r for r in rules.get(cid, []) if r['type'] in ['data', 'bitmask'] and f"RuleMatch: \"{r['message']}\"" in alert_reasons), {'type':'unknown'})
                    packet_hex = format_packet_for_alert(data, triggering_rule)
                    reason_str = " | ".join(alert_reasons)
                    log_line = f"*** ALERT [{t_str}]: ID 0x{cid:03X} Len:{data_len} Data: {packet_hex} Reason: {reason_str} ***\n"

                    if print_to_console: print(log_line, end='')
                    log_file.write(log_line)
                    log_file.flush() # Write immediately

    except KeyboardInterrupt:
        print("\n[INFO] Stopping Hybrid IDS monitoring.")
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred in main loop: {e}")

# --- Argument Parsing & Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid CAN Bus Intrusion Detection System (Signature + Anomaly)")
    parser.add_argument("-c", "--channel", default="vcan0", help="CAN interface name (e.g., vcan0, can0)")
    parser.add_argument("-r", "--rules", default="can_rules.txt", help="Path to the signature rule file")
    parser.add_argument("-m", "--models", default=DEFAULT_MODEL_DIR, help=f"Directory containing trained ML models/scalers (default: {DEFAULT_MODEL_DIR})")
    parser.add_argument("-l", "--log", default="hybrid_alerts.log", help="Path to the alert log file")
    parser.add_argument("-p", "--print", action="store_true", help="Print alerts to console in addition to logging")
    parser.add_argument("--test-rules", action="store_true", help="Test the rule file syntax and exit")
    args = parser.parse_args()

    if args.test_rules:
        print("[INFO] Testing rule file syntax...")
        parse_rules(args.rules) # Just run the parser to check rules
        print("[INFO] Rule file test complete. Exiting.")
    else:
        run_hybrid_ids(args.rules, args.models, args.channel, args.log, args.print)