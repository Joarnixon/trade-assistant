import os
from Coefficients import Static
def create_stock_folders_and_files(stock_names, folder):
    parent_folder = folder
    os.makedirs(parent_folder, exist_ok=True)

    for stock_name in stock_names:
        folder_path = os.path.join(os.getcwd(), parent_folder, stock_name)
        os.makedirs(folder_path, exist_ok=True)

        log_files = ['PriceLog', 'BuyersLog', 'SellersLog', 'BidsLog', 'AsksLog', 'WeightedBidLog', 'WeightedAskLog', 'PricesLog']
        for log_file in log_files:
            file_path = os.path.join(folder_path, f'{log_file}.txt')
            open(file_path, 'w').close()

def read_stock_log(stock_name, log_file, folder):
    parent_folder = folder
    folder_path = os.path.join(os.getcwd(), parent_folder, stock_name)
    file_path = os.path.join(folder_path, f'{log_file}.txt')
    with open(file_path, 'r') as file:
        content = file.read()
    return content

def write_stock_log(stock_name, log_file, folder, content, mode='a'):
    parent_folder = folder
    folder_path = os.path.join(os.getcwd(), parent_folder, stock_name)
    file_path = os.path.join(folder_path, f'{log_file}.txt')
    with open(file_path, mode) as file:
        file.write(content)
# Example usage
def removal(stock_name, log_file, folder):
    parent_folder = folder
    for stock_name in stock_name:
        folder_path = os.path.join(os.getcwd(), parent_folder, stock_name)
        file_path = os.path.join(folder_path, f'{log_file}.txt')
        os.remove(file_path)

stock_names = Static.Stock
create_stock_folders_and_files(stock_names, 'StocksLog1')
#removal(stock_names, 'BidsPriceLog')



