import csv
import numpy as np
max_rows = 2000
rows_written = 0

lip_coords = np.ones((40, 3))




while True:
  if rows_written < max_rows:
  # with open('output.csv', 'a', newline='') as csvfile:
  #     csv_writer = csv.writer(csvfile)
  #     # Write the list as a row in the CSV file
  
    np.savetxt('output.csv', lip_coords, delimiter=',')
            # csv_writer.writerow(lip_coord)
    rows_written += 1