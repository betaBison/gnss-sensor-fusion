import csv
import pandas as pd

# Extract data different timesteps
def Extract_timedata(input_path):
  raw_data = []
  fix_data = []
  header_fix = ''
  header_raw = ''
  with open(input_path, 'r') as f:
    t = -1
    for line in f:
      if line[0] == '#':
        line_data = line[2:].rstrip('\n').replace(" ","").split(",")
        if line_data[0] == 'Raw':
          header_raw = line_data[1:]
        elif line_data[0] == 'Fix':
          header_fix = line_data[1:]
        continue
      line_data = line.rstrip('\n').replace(" ","").split(",")
      if line_data[0] == 'Fix':
        fix_data.append(line_data[1:])
        raw_data.append([])
        t += 1
      elif line_data[0] == 'Raw':
        raw_data[t].append(line_data[1:])
  return header_raw, raw_data, header_fix, fix_data


# Extract data continuous and make a csv
def MakeCsv(input_path):
  out_path = "./data/raw.csv"
  with open(out_path, 'w') as f:
    writer = csv.writer(f)
    with open(input_path, 'r') as f:
      for line in f:
        # Comments in the log file
        if line[0] == '#':
          # Remove initial '#', spaces, trailing newline and split using commas as delimiter
          line_data = line[2:].rstrip('\n').replace(" ","").split(",")
          if line_data[0] == 'Raw':
            writer.writerow(line_data[1:])
        # Data in file
        else:
          # Remove spaces, trailing newline and split using commas as delimiter
          line_data = line.rstrip('\n').replace(" ","").split(",")
          if line_data[0] == 'Raw':
            writer.writerow(line_data[1:])
  return out_path


def MakeCsvFix(input_path):
  out_path = "./data/fix.csv"
  with open(out_path, 'w') as f:
    writer = csv.writer(f)
    with open(input_path, 'r') as f:
      for line in f:
        # Comments in the log file
        if line[0] == '#':
          # Remove initial '#', spaces, trailing newline and split using commas as delimiter
          line_data = line[2:].rstrip('\n').replace(" ","").split(",")
          if line_data[0] == 'Fix':
            writer.writerow(line_data[1:])
        # Data in file
        else:
          # Remove spaces, trailing newline and split using commas as delimiter
          line_data = line.rstrip('\n').replace(" ","").split(",")
          if line_data[0] == 'Fix':
            writer.writerow(line_data[1:])
  return out_path
