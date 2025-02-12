from GenerateAndSerialise import find_month_numbers, load_data_structure, parsing_specific
from datetime import datetime, timedelta
import re
import math
import time

loaded_expiries = load_data_structure('all_contract_expiries.pickle')

def parsing_generic(generic_code):
    """
    Returns: A generic code parsed into the associated product, and n, where n
    is the number of active contracts away the given contract is from the input date
    """
    match = re.match(r"([A-Z]+)(\d+)$", generic_code)
    prdct_code, gnric_number = match.groups()
    return prdct_code, int(gnric_number)

def parsing_mthlygeneric(monthly_generic_code):
    """
    Returns: A monthly generic code parsed into the associated product, its
    delivery month number and n where n is the nth closest contract of this
    delivery month
    """
    match = re.match(
        r"([A-Z]+)(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(\d+)$",
        monthly_generic_code)
    prdct_code, mnth, sequence_number = match.groups()
    return prdct_code, mnth, int(sequence_number)

def date_range(start_date, end_date):
    """
    A function that generates the date range to be used for iterating through
    the point-in-time input dates when mapping securities
    """
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(n)


def get_month_number(month_abbreviation):
    """
    Returns: A month number corresponding to a delivery month 3 letter code,
    i.e. 1 for JAN, 2 for FEB, 3 for MAR etc.
    """
    months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
              "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    month_to_number = {month: i + 1 for i, month in enumerate(months)}
    return month_to_number[month_abbreviation]

def get_month_letter(number):
    """
    Returns: the delivery month letter corresponding to a month number. E.g. 1
    for F which is January, 2 for G which is February etc.
    """
    letters = ["F", "G", "H", "J", "K", "M",
              "N", "Q", "U", "V", "X", "Z"]
    return letters[number - 1]


def querying_interfaceStoG(security, output_type, date_range_str):
    """
    Child function of the parent function querying_interface(security,
    input_type, date_range_str)
    :param security: the input security of specific form
    :param output_type: 'generic'
    :param date_range_str: the range of dates for which we map the specific
    Returns dict of {input date : generic form for given date} - the dictionary
    grows for larger date ranges. None values if the contract has expired.
    """
    if output_type != 'generic':
        return "not generic"

    if '-' in date_range_str:
        start_date_str, end_date_str = date_range_str.split('-')
    else:
        start_date_str = end_date_str = date_range_str
    start_date = datetime.strptime(start_date_str, '%Y/%m/%d').date()
    end_date = datetime.strptime(end_date_str, '%Y/%m/%d').date()

    product_code, month_letter, target_del_year = parsing_specific(security)
    month_numbers = find_month_numbers()
    target_del_month = month_numbers[month_letter]

    target_expiry_date = loaded_expiries[product_code][target_del_year] \
        [target_del_month]['expiry_date']

    def extract_expiry_dates(expiries, product):
        expiry_date_set = set()
        for year in expiries[
            product].values():
            for month in year.values():
                expiry_date_set.add(month['expiry_date'])
        return expiry_date_set

    all_expiry_dates = extract_expiry_dates(loaded_expiries, product_code)

    results = {}
    for single_date in date_range(start_date, end_date):
        input_date = single_date

        generic_number = 0
        if input_date < target_expiry_date:
            relevant_expiries = set()
            current_date = input_date
            while current_date <= target_expiry_date:
                next_date = current_date + timedelta(days=1)
                if next_date in all_expiry_dates:
                    relevant_expiries.add(next_date)
                    generic_number += 1
                current_date += timedelta(days=1)

        results[single_date.strftime(
            '%Y-%m-%d')] = f"{product_code}{generic_number}" if generic_number != 0 else None
    return results



def querying_interfaceGtoS(security, output_type, date_range_str):
    """
    Child function of the parent function querying_interface(security,
    input_type, date_range_str)
    :param security: the input security of generic form
    :param output_type: 'specific'
    :param date_range_str: the range of dates for which we map the generic
    Returns dict of {input date : specific form for input date} - the dictionary
    grows for larger date ranges. None values if the contract has expired.
    """
    if output_type != 'specific':
        return "not specific"

    if '-' in date_range_str:
        start_date_str, end_date_str = date_range_str.split('-')
    else:
        start_date_str = end_date_str = date_range_str
    start_date = datetime.strptime(start_date_str, '%Y/%m/%d').date()
    end_date = datetime.strptime(end_date_str, '%Y/%m/%d').date()

    product_code, generic_number = parsing_generic(security)
    base_generic_number = generic_number

    def extract_expiry_dates(expiries, product):
        expiry_date_set = set()
        for year in expiries[
            product].values():
            for month in year.values():
                expiry_date_set.add(month['expiry_date'])
        return expiry_date_set

    all_expiry_dates = extract_expiry_dates(loaded_expiries, product_code)
    results = {}
    for single_date in date_range(start_date, end_date):
        current_date = single_date
        relevant_expiries = []
        generic_number = base_generic_number
        while generic_number >= 1:
            next_date = current_date + timedelta(days=1)
            if next_date in all_expiry_dates:
                relevant_expiries.append(next_date)
                generic_number -= 1
            current_date += timedelta(days=1)

        def find_delivery_yearmonth(expiries, prdct_code, target_expiry_date):
            for year, months in expiries[prdct_code].items():
                for month, expiry_dates in months.items():
                    if expiry_dates['expiry_date'] == target_expiry_date:
                        return year, month

        target_del_year, target_del_month =\
            find_delivery_yearmonth(loaded_expiries, product_code,
                                    relevant_expiries[-1])

        stryeardigits = str(target_del_year)[-2:]
        delivery_letter = get_month_letter(target_del_month)

        results[single_date.strftime('%Y-%m-%d')] = \
            f"{product_code}{delivery_letter}{stryeardigits}"

    return results



def querying_interfaceGtoMG(security, output_type, date_range_str):
    """
    Child function of the parent function querying_interface(security,
    input_type, date_range_str)
    :param security: the input security of generic form
    :param output_type: 'monthly generic'
    :param date_range_str: the range of dates for which we map the generic
    Returns dict of {input date : MG form for input date} - the dictionary
    grows for larger date ranges. Returns None values for expired contracts
    """
    if output_type != 'monthly generic':
        return "not monthly generic"

    if '-' in date_range_str:
        start_date_str, end_date_str = date_range_str.split('-')
    else:
        start_date_str = end_date_str = date_range_str
    start_date = datetime.strptime(start_date_str, '%Y/%m/%d').date()
    end_date = datetime.strptime(end_date_str, '%Y/%m/%d').date()

    product_code, generic_number = parsing_generic(security)
    monthly_generic_number = math.ceil(generic_number / 12)

    def extract_expiry_dates(expiries, product):
        expiry_date_set = set()
        for year in expiries[
            product].values():
            for month in year.values():
                expiry_date_set.add(month['expiry_date'])
        return expiry_date_set

    all_expiry_dates = extract_expiry_dates(loaded_expiries, product_code)
    results = {}
    for single_date in date_range(start_date, end_date):
        current_date = single_date
        next_date = current_date + timedelta(days=1)
        while next_date not in all_expiry_dates:
            current_date = next_date
            next_date = current_date + timedelta(days=1)

        def find_delivery_month(expiries, prdct_code, target_expiry_date):
            for year, months in expiries[prdct_code].items():
                for month, expiry_date in months.items():
                    if expiry_date['expiry_date'] == target_expiry_date:
                        return month

        input_exptodel_month = find_delivery_month(loaded_expiries,
                                                   product_code,
                                                   next_date)

        month_number = input_exptodel_month + (generic_number - 1) % 12
        if month_number > 12:
            month_number -= 12

        def get3lettermonth(month_number):
            months = 'JANFEBMARAPRMAYJUNJULAUGSEPOCTNOVDEC'
            return months[3 * (month_number - 1):3 * (month_number - 1) + 3]

        month3letter = get3lettermonth(month_number)

        results[single_date.strftime('%Y-%m-%d')] = \
            f"{product_code}{month3letter}{monthly_generic_number}" \
                if monthly_generic_number != 0 else None

    return results



def querying_interfaceMGtoG(security, output_type, date_range_str):
    """
    Child function of the parent function querying_interface(security,
    input_type, date_range_str)
    :param security: the input security of monthly generic form
    :param output_type: 'generic'
    :param date_range_str: the range of dates for which we map the generic
    Returns dict of {input date : MG form for input date} - the dictionary
    grows for larger date ranges. Returns None values for expired contracts
    """
    if output_type != 'generic':
        return "not generic"

    if '-' in date_range_str:
        start_date_str, end_date_str = date_range_str.split('-')
    else:
        start_date_str = end_date_str = date_range_str
    start_date = datetime.strptime(start_date_str, '%Y/%m/%d').date()
    end_date = datetime.strptime(end_date_str, '%Y/%m/%d').date()

    product_code, delivery_month_str, monthly_generic_number =\
        parsing_mthlygeneric(security)
    delivery_month = get_month_number(delivery_month_str)

    def extract_expiry_dates(expiries, product):
        expiry_date_set = set()
        for year in expiries[
            product].values():
            for month in year.values():
                expiry_date_set.add(month['expiry_date'])
        return expiry_date_set
    all_expiry_dates = extract_expiry_dates(loaded_expiries, product_code)

    results = {}
    for single_date in date_range(start_date, end_date):
        current_date = single_date
        next_date = current_date + timedelta(days=1)
        while next_date not in all_expiry_dates:
            current_date = next_date
            next_date = current_date + timedelta(days=1)

        def find_delivery_month(expiries, prdct_code, target_expiry_date):
            for year, months in expiries[prdct_code].items():
                for month, expiry_date in months.items():
                    if expiry_date['expiry_date'] == target_expiry_date:
                        return month
        input_delivery_month = find_delivery_month(loaded_expiries, product_code,
                                             next_date)

        generic_number = 12 * (monthly_generic_number - 1) + (
                    delivery_month - input_delivery_month + 1)
        #adjust for year wrap around
        if input_delivery_month > delivery_month or generic_number < 0:
            generic_number += 12


        results[single_date.strftime('%Y-%m-%d')] =\
            f"{product_code}{generic_number}"\
                if generic_number != 0 else None

    return results

#print(querying_interfaceMGtoG('COJAN3', 'generic', '2025/01/21-2025/01/23'))
#date range
#print(querying_interfaceMGtoG('CL12', 'generic', '2025/01/21'))


def querying_interfaceStoMG(security, output_type, date_range_str):
    """
    Child function of the parent function querying_interface(security,
    input_type, date_range_str). Uses S->G->MG pipeline for computation
    :param security: the input security of specific form
    :param output_type: 'monthly generic'
    :param date_range_str: the range of dates for which we map the generic
    Returns dict of {input date : MG form for input date} - the dictionary
    grows for larger date ranges. Returns None values for expired contracts
    """
    StoGdict = querying_interfaceStoG(security, 'generic', date_range_str)
    result = {}
    for date, product in StoGdict.items():
        if product == None:
            result[date] = product
        else:
            date = date.replace('-', '/')
            currentdatedict = querying_interfaceGtoMG(product, output_type, date)
            result.update(currentdatedict)
    return result

def querying_interfaceMGtoS(security, output_type, date_range_str):
    """
    Child function of the parent function querying_interface(security,
    input_type, date_range_str). Uses MG->G->S pipeline for computation
    :param security: the input security of specific form
    :param output_type: 'monthly generic'
    :param date_range_str: the range of dates for which we map the generic
    Returns dict of {input date : MG form for input date} - the dictionary
    grows for larger date ranges. Returns None values for expired contracts
    """
    MGtoGdict = querying_interfaceMGtoG(security, 'generic', date_range_str)
    result = {}
    for date, product in MGtoGdict.items():
        if product == None:
            result[date] = product
        else:
            date = date.replace('-', '/')
            currentdatedict = querying_interfaceGtoS(product, output_type, date)
            result.update(currentdatedict)
    return result


def querying_interface(security, input_type, date_range_str):
    """
    Parent function used to query the data structure from part 1
    :param security: the input security of any form
    :param input_type: the type of the input security
    :param date_range_str: the range of dates for which we map the security
    Returns a tuple of dicts of the forms we are mapping to.
    {input date : security1 for input}{input date : security2 for input}
    """
    dict_of_types = {'specific': 'S', 'generic':'G', 'monthly generic':'MG'}
    set_of_types = set()
    for type in dict_of_types:
        if type != input_type:
            set_of_types.add(dict_of_types[type])
    if 'S' in set_of_types and 'G' in set_of_types:
        gneric_dict = querying_interfaceMGtoG(security, 'generic', date_range_str)
        spcific_dict = querying_interfaceMGtoS(security, 'specific', date_range_str)
        return spcific_dict, gneric_dict
    elif 'S' in set_of_types and 'MG' in set_of_types:
        spcific_dict = querying_interfaceGtoS(security, 'specific', date_range_str)
        mthlygneric_dict = querying_interfaceGtoMG(security, 'monthly generic', date_range_str)
        return spcific_dict, mthlygneric_dict
    else:
        gneric_dict = querying_interfaceStoG(security, 'generic', date_range_str)
        mthlygneric_dict = querying_interfaceStoMG(security, 'monthly generic', date_range_str)
        return gneric_dict, mthlygneric_dict


#start = time.time()
#print(querying_interface('COJAN2', 'monthly generic', '2024/11/28'))
#end = time.time()
#print(end - start)


