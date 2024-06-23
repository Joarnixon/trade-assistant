import datetime
from MeanVolume import VolumeOperations
import asyncio
import ast
import os
from Coefficients import Dynamic
volume_op = VolumeOperations()


class Handling:

    def write_log(self, mean_volume):
        current_time = datetime.datetime.now()
        with open("log.txt", "w") as log_file:
            log_file.write(current_time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
            log_file.write(str(mean_volume))

    def read_log(self):
        with open("log.txt", "r") as log_file:
            lines = log_file.readlines()
            if len(lines) < 2:
                return None, None
            else:
                timestamp = datetime.datetime.strptime(lines[0].strip(), "%Y-%m-%d %H:%M:%S")
                value = lines[1]
                return timestamp, value

    def read_alerts(self):
        with open("alerts.txt", "r") as log_file:
            content = log_file.read()

            # Check if content is empty and return an empty list if it is
            if not content:
                return []

            # Parse the string representation of the list directly
            stock_symbols = ast.literal_eval(content)
            return stock_symbols

    def write_alerts(self, figis):
        with open("alerts.txt", "w") as log_file:
            log_file.write(str(figis))

    def delete_alerts(self):
        with open("alerts.txt", "w") as log_file:
            log_file.write("")

    def handle_log(self):
        last_execution_time = Handling.read_log(self)[0]

        if last_execution_time is None or (datetime.datetime.now() - last_execution_time).days >= 2:
            print('...Please wait for average-volume calculations...')
            mean_volume = asyncio.run(VolumeOperations.main(self=volume_op))
            Handling.write_log(self, mean_volume)
            print('...Done!...')
            Dynamic.MV = mean_volume
        else:
            time_since_last_execution = datetime.datetime.now() - last_execution_time
            time_until_next_execution = datetime.timedelta(days=2) - time_since_last_execution
            days_remaining = time_until_next_execution.days
            hours_remaining = time_until_next_execution.seconds // 3600
            print(f"Time until next execution: {days_remaining} Days:{hours_remaining} Hours")
            Dynamic.MV = ast.literal_eval(Handling.read_log(self)[1])

    def read_stock_log(self, folder, figi, log_file):
        parent_folder = folder
        folder_path = os.path.join(os.getcwd(), parent_folder, figi)
        file_path = os.path.join(folder_path, f'{log_file}.txt')
        with open(file_path, 'r') as file:
            content = file.read()
        return content

    def write_stock_log(self, figi, log_file, folder, content, mode='a'):
        parent_folder = folder
        folder_path = os.path.join(os.getcwd(), parent_folder, figi)
        file_path = os.path.join(folder_path, f'{log_file}.txt')
        with open(file_path, mode) as file:
            file.write(content)

    def write_trend_log(self, log_file, folder, content, mode='a'):
        parent_folder = folder
        folder_path = os.path.join(os.getcwd(), parent_folder)
        file_path = os.path.join(folder_path, f'{log_file}.txt')
        with open(file_path, mode) as file:
            file.write(content)
