import pandas as pd
import pickle

def find_month_numbers():
    """
    Returns: dict of {delivery month letter: month number} mappings
    """
    letters = 'FGHJKMNQUVXZ'
    numbers = range(1, 13)
    month_codes = {letter: number for letter, number in zip(letters,
            numbers)}
    return month_codes

def parsing_specific(specific_code):
    """
    Returns: A specific code parsed into product, delivery month letter, year
    """
    prdct_code = specific_code[:-3]
    mth_letter = specific_code[-3]
    yr = '20' + specific_code[-2:]
    return prdct_code, mth_letter, int(yr)

def create_data_structure(contracts_expiries):
    """
    Creates a nested dictionary from a CSV file that organises contract expiries
    by product code, year, and month.

    Args:
    contracts_expiries (str): File path to the CSV containing expiry data
    with columns 'specific', 'expiry_date', and 'delivery_month'.

    Returns: dict with a multi-level hierarchy based on product code, year
    and month of delivery and its value being expiry date
    """
    expiry_calendars = pd.read_csv(contracts_expiries)
    expiry_calendars['expiry_date'] = pd.to_datetime(
        expiry_calendars['expiry_date']).dt.date
    expiry_calendars['delivery_month'] = pd.to_datetime(
        expiry_calendars['delivery_month']).dt.date

    contracts = {}
    month_numbers = find_month_numbers()

    for index, row in expiry_calendars.iterrows():
        specific = row['specific']
        product_code, month_letter, year = parsing_specific(specific)
        month_number = month_numbers[month_letter]

        contracts.setdefault(product_code, {}).setdefault(year, {}).setdefault(
            month_number, {
                'expiry_date': row['expiry_date'],
            })

    return contracts

def save_data_structure(data, filepath):
    """
    Serialises the given data structure and saves it to a file.
    """
    with open(filepath, 'wb') as file:
        pickle.dump(data, file)

def load_data_structure(filepath):
    """
    Loads a serialised data structure from a file.
    """
    with open(filepath, 'rb') as file:
        return pickle.load(file)


all_contract_expiries = create_data_structure("expiry_calendars.csv")
save_data_structure(all_contract_expiries, 'all_contract_expiries.pickle')
