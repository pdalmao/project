import can
import time
import os
import datetime
import numpy as np
import operator as op
import argparse
from collections import deque, defaultdict

# --- Constants ---
FREQ_WINDOW_SIZE = 10 # Number of messages to calculate frequency with
MIN_MSGS_FOR_FREQ = 2 # Min number of messages to determine frequency
RULE_RELOAD_INTERVAL = 60  # Seconds - How long until rules hot reload

# --- Frequency Calculation ---
def calculate_frequency(timestamps): 
    if len(timestamps) < MIN_MSGS_FOR_FREQ:
        return None
    intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1) if timestamps[i+1] - timestamps[i] > 0]
    if not intervals:
        return None # If timestamps cannot be calculated for whatever reason, return None to not evoke false positives
    median_interval = np.median(intervals)
    return float('inf') if median_interval <= 0 else 1.0 / median_interval

# --- Rule Parsing ---
def parse_rules(filepath):
    invalid = 0
    rules = defaultdict(list) # Use defaultdict to automatically create lists for new CAN IDs
    if not os.path.exists(filepath):
        print(f"[ERROR] Rule file not found: {filepath}")
        return None
    else:
        with open(filepath, 'r') as f:
            num_rules = sum(1 for line in f if not line.startswith('#')) # Not entirely sure why I have to put this here instead of in the try function below.
    try:
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1): # Formatting lines and making sure comments aren't passed into the parsing algorithm
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split() # Break up rules into their constituent elements for parsing
                try:
                    # --- Part 1 (Alert type) Parsing ---
                    # Checking for type of alert and making sure it is valid
                    rule_type = parts[0].upper()
                    if rule_type == 'ALERT':
                        can_id_str, byte_spec_str, op1, *rest = parts[1:]
                        if '-' in can_id_str:
                            start_id_str, end_id_str = can_id_str.split('-')
                            start_id = int(start_id_str, 0)
                            end_id = int(end_id_str, 0)
                            can_ids = range(start_id, end_id + 1)
                        else:
                            can_ids = [int(can_id_str, 0)]


                        # --- Part 2 (Operator) Parsing ---
                        # Check for byte range and parse if there is a hyphen present (e.g. bytes 3-4)
                        if '-' in byte_spec_str:
                            byte_spec = tuple(int(b) for b in byte_spec_str.split('-'))
                        else:
                            byte_spec = int(byte_spec_str)

                        # Bitmask detection (e.g. &0xF0)
                        if op1.startswith('&'):
                            mask = int(op1[1:], 0)
                            operator = rest[0]
                            value = int(rest[1], 0)
                            message = ' '.join(rest[2:])
                            rules[can_id].append({
                                'type': 'bitmask',
                                'byte_index': byte_spec if isinstance(byte_spec, int) else byte_spec[0],
                                'mask': mask,
                                'operator': operator,
                                'value': value,
                                'message': message
                            })
                            continue

                        operator = op1.lower()
                        value_str = rest[0]
                        message = ' '.join(rest[1:])
                        # Value parsing
                        if operator in ['==', '!=', '>', '<', '>=', '<=']:
                            value = int(value_str, 0)
                        elif operator in ['in', 'notin']:
                            value = set(int(v, 0) for v in value_str.split(',')) # If operator is "in" or "notin" then check all comma-delimited values
                        else:
                            raise ValueError("Unsupported operator")

                        # --- Part 3 (Information) Parsing ---
                        for can_id in can_ids:
                            rules[can_id].append({
                                'type': 'data',
                                'byte_spec': byte_spec,
                                'operator': operator,
                                'value': value,
                                'message': message
                            })

                    # Only Part 3 parsing needed for frequency requests as the only relevant info is the frequency itself
                    elif rule_type == 'FREQ_ALERT':
                        can_id_str = parts[1]
                        min_freq = float(parts[2])
                        max_freq = float(parts[3])
                        message = ' '.join(parts[4:])

                        # Support ID range
                        if '-' in can_id_str:
                            start_id_str, end_id_str = can_id_str.split('-')
                            start_id = int(start_id_str, 0)
                            end_id = int(end_id_str, 0)
                            can_ids = range(start_id, end_id + 1)
                        else:
                            can_ids = [int(can_id_str, 0)]

                        for can_id in can_ids:
                            rules[can_id].append({
                                'type': 'frequency',
                                'min_freq': min_freq,
                                'max_freq': max_freq,
                                'message': message
                            })

                    elif rule_type == 'LENGTH_ALERT':
                        can_id_str = parts[1]
                        max_length = int(parts[2])
                        message = ' '.join(parts[3:])

                        # Support ID range
                        if '-' in can_id_str:
                            start_id_str, end_id_str = can_id_str.split('-')
                            start_id = int(start_id_str, 0)
                            end_id = int(end_id_str, 0)
                            can_ids = range(start_id, end_id + 1)
                        else:
                            can_ids = [int(can_id_str, 0)]

                        for can_id in can_ids:
                            rules[can_id].append({
                                'type': 'length',
                                'max_length': max_length,
                                'message': message
                            })
                        
                # Use the invalid variable to check for however many invalid rules there are, and then show the user how many valid rules there are vs total.
                except Exception as e:
                    print(f"[WARN] Skipping invalid rule on line {line_num}: \"{line}\" - {e}")
                    invalid += 1
                    continue
        print(f"[INFO] Loaded {num_rules - invalid} valid rules out of {num_rules} from {filepath}")
    except Exception as e:
        print(f"[ERROR] Failed to read rule file: {e}")
        return None
    return rules

# --- Rule Evaluators ---
def evaluate_data_rule(data, rule):
    op_map = {'==': op.eq, '!=': op.ne, '>': op.gt, '<': op.lt, '>=': op.ge, '<=': op.le, 'in': lambda a, b: a in b, 'notin': lambda a, b: a not in b} # Make map of operators so they're all readily available for evaluation
    index = rule['byte_spec']
    if isinstance(index, tuple): # If index is a tuple, it's a byte range (e.g., "3-4")
        start, end = index
        if end >= len(data): return False
        value = int.from_bytes(data[start:end+1], byteorder='big')
    else:
        if index >= len(data): return False
        value = data[index]
    return op_map[rule['operator']](value, rule['value']) # Evaluates if rule is triggered with comparison

def evaluate_bitmask_rule(data, rule):
    op_map = {'==': op.eq, '!=': op.ne, '>': op.gt, '<': op.lt, '>=': op.ge, '<=': op.le}
    index = rule['byte_index']
    if index >= len(data):
        return False
    masked_value = data[index] & rule['mask']
    return op_map[rule['operator']](masked_value, rule['value']) # See above

# --- Main Function ---
def filter_can_data_with_rules(rule_file, channel, log_file_path, print_to_console):
    # Check if testing flag is test and if it is call the "parse_rules" function to display which rules are valid
    if args.test:
        print("[INFO] Testing rule file...")
        parse_rules(rule_file)
        print("[INFO] Test is complete. Exiting.")
        return
    
    # --- Rule Loading Logic --- 
    rules = parse_rules(rule_file)
    if rules is None:
        return
    timestamps = defaultdict(lambda: deque(maxlen=FREQ_WINDOW_SIZE))
    last_reload_time = time.time()

    with open(log_file_path, "a") as log_file:
        try:
            # Begin reading from bus
            with can.interface.Bus(channel=channel, interface='socketcan') as bus:
                if print_to_console:
                    print(f"Monitoring CAN on {channel}...")
                while True:
                    # Check the time difference using the RULE_RELOAD_INTERVAL constant as the No. seconds
                    if time.time() - last_reload_time > RULE_RELOAD_INTERVAL:
                        rules = parse_rules(rule_file)
                        last_reload_time = time.time()

                    # Recieve messages
                    msg = bus.recv(1.0)
                    if msg is None: continue # This stops the program from crashing when there is no message recieved on the bus after the timeout period
                    now = time.time()
                    cid = msg.arbitration_id
                    timestamps[cid].append(now)
                    data = msg.data

                    # --- Rule Comparison ---
                    for rule in rules.get(cid, []):
                        alert = False
                        if rule['type'] == 'frequency':
                            freq = calculate_frequency(timestamps[cid])
                            if freq is not None and not (rule['min_freq'] <= freq <= rule['max_freq']):
                                alert = True
                        elif rule['type'] == 'data':
                            alert = evaluate_data_rule(data, rule)
                        elif rule['type'] == 'bitmask':
                            alert = evaluate_bitmask_rule(data, rule)
                        elif rule['type'] == 'length':
                            if len(data) > rule['max_length']:
                                alert = True

                        # --- Alert Logic ---
                        if alert:
                            t = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                            def format_packet(data, rule):
                                # Create formatted hex with matched bytes/nibbles marked
                                parts = []
                                target_index = -1
                                mask = 0
                                highlight_indices = set() # For data rules

                                # Determine which indices/byte to highlight based on rule type
                                if rule['type'] == 'data':
                                    bs = rule['byte_spec']
                                    if isinstance(bs, tuple):
                                        # Ensure the range is valid before adding to highlight
                                        start, end = bs
                                        # Check bounds to prevent IndexError if rule specifies bytes beyond data length
                                        if start < len(data) and end < len(data):
                                            highlight_indices.update(range(start, end + 1))
                                    elif isinstance(bs, int):
                                        if bs < len(data): # Check bounds for single index
                                            highlight_indices.add(bs)
                                elif rule['type'] == 'bitmask':
                                    if rule['byte_index'] < len(data): # Check bounds for bitmask index
                                        target_index = rule['byte_index']
                                        mask = rule['mask'] # Store the mask for later use

                                # Iterate through the data bytes and format them
                                for i, byte in enumerate(data):
                                    hex_byte = f"{byte:02X}"

                                    if i == target_index: # Handle bitmask highlighting (nibble-specific)
                                        high_nibble = hex_byte[0]
                                        low_nibble = hex_byte[1]
                                        # Check which nibbles the mask affects
                                        high_mask_bits = mask & 0xF0
                                        low_mask_bits = mask & 0x0F

                                        # --- Nibble Highlighting Logic ---
                                        if high_mask_bits != 0 and low_mask_bits == 0:
                                            # High nibble logic
                                            parts.append(f"[{high_nibble}]{low_nibble}")
                                        elif low_mask_bits != 0 and high_mask_bits == 0:
                                            # Low nibble logic
                                            parts.append(f"{high_nibble}[{low_nibble}]")
                                        elif high_mask_bits != 0 and low_mask_bits != 0:
                                            # If mask affects both nibbles then highlight entire byte
                                            parts.append(f"[{hex_byte}]")
                                        else:
                                            # If mask doesn't apply for whatever reason then don't highlight anything
                                            parts.append(hex_byte)

                                    elif i in highlight_indices: # Handle data rule highlighting (whole byte)
                                        parts.append(f"[{hex_byte}]")
                                    else: # No highlighting for this byte
                                        parts.append(hex_byte)
                                return ' '.join(parts)

                            packet_hex = format_packet(data, rule) # Get formatted hex string for the alert
                            # Logic to replace the {freq} element that could be found in a rule
                            msg_text = rule['message']
                            if '{freq}' in msg_text:
                                msg_text = msg_text.replace("{freq}", f"{freq:.2f}Hz")
                            if '{length}' in msg_text:
                                msg_text = msg_text.replace("{length}", str(len(data)))
                            log_line = f"*** ALERT [{t}]: ID 0x{cid:X}    {packet_hex}    \"{msg_text}\" ***\n" # Assemble message and format it so it looks nice :)

                            if print_to_console: # If the print flag is set then print the alert to the console
                                print(log_line, end='')
                            log_file.write(log_line) # Write the alert to the log file
                            log_file.flush() # Ensures that the alert is written without delay
        except KeyboardInterrupt:
            print("\nStopping monitoring.") # Graceful exit
        except Exception as e:
            print(f"An unexpected error occurred: {e}") # Shows the error in a user-friendly format instead of the ugly default python formatting

# --- Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--channel", default="vcan0", help="CAN interface (e.g., vcan0)")
    parser.add_argument("-r", "--rules", default="can_rules.txt", help="Path to the rule file")
    parser.add_argument("-l", "--log", default="alerts.log", help="Path to the log file")
    parser.add_argument("-p", "--print", action="store_true", help="Print alerts to console")
    parser.add_argument("-t", "--test", action="store_true", help="Tests the rule file to see which rules are valid")
    args = parser.parse_args()

    # Call the main function
    filter_can_data_with_rules(args.rules, args.channel, args.log, args.print)
