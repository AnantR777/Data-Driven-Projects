import LoadAndQuery as lq
import GenerateAndSerialise as gs

loaded_expiries = lq.load_data_structure('all_contract_expiries.pickle')

def extract_monthly_gen(product_code, tuple):
    """
    :param product_code: e.g. CO OR CL
    :param tuple: the (monthA, monthB) spread
    Returns 3 letter month codes and monthly generics to get the spread
    """
    mnthA, mnthB = tuple[0], tuple[1]
    mnthly_gnericA = product_code + mnthA + "1"
    mnthly_gnericB = product_code + mnthB + "1"
    return mnthA, mnthB, mnthly_gnericA, mnthly_gnericB

def getRelevantCodes(mnthly_gnericA, mnthly_gnericB, mnthB, single_date):
    """
    :param mnthly_gnericA: monthly generic generated from first tuple element
    :param mnthly_gnericB: monthly generic generated from second tuple element
    :param mnthB: second month used to get the second part of the spread
    :param single_date: date used to determine generic/monthlygen spread code
    :return: the spread in a potentially [specific, not-specific] form and a
    boolean for if the spread is in [specific, specific] form or not
    """
    specific_dictA, _ = \
        lq.querying_interface(mnthly_gnericA, 'monthly generic', single_date)
    specific_dictB, _ = \
        lq.querying_interface(mnthly_gnericB, 'monthly generic', single_date)

    single_date = single_date.replace('/', '-')
    specificA = specific_dictA[single_date]
    specificB = specific_dictB[single_date]

    product_code, del_monthA, yearA = gs.parsing_specific(specificA)
    monthA_number = gs.find_month_numbers()[del_monthA]
    product_code, del_monthB, yearB = gs.parsing_specific(specificB)
    monthB_number = gs.find_month_numbers()[del_monthB]

    first_monthA_expdate = loaded_expiries[product_code][yearA][monthA_number][
        'expiry_date']
    first_monthB_expdate = loaded_expiries[product_code][yearB][monthB_number][
        'expiry_date']
    if first_monthA_expdate < first_monthB_expdate:
        spread = [specificA, specificB]
        AbforeB = True
    else:
        spread = [specificA, product_code + mnthB +"2"]
        AbforeB = False

    return spread, AbforeB


def convert_to_specific(product_code, spread, mthA, mthB, AbforeB, single_date):
    """
    Converts a [specific, not-specific] spread into [specific, specific] by
    converting the second 'not-specific' element if necessary
    """
    older_product, newer_product = spread[0], spread[1]
    if AbforeB:
        return {(product_code+mthA+mthB+"1", product_code+mthA+"1"+mthB+"1"):
            [older_product, newer_product]}
    else:
        spec_dictB2, _ = \
            lq.querying_interface(newer_product, 'monthly generic',
                                  single_date)
        single_date = single_date.replace('/', '-')
        newer_product = spec_dictB2[single_date]
        return {(product_code+mthA+mthB+"1", product_code+mthA+"1"+mthB+"2"):
            [older_product, newer_product]}

def get_spreads(product_code, list_of_tuples, single_date):
    """
    Uses functions above to take a list of tuples and output a dictionary
    consisting of {(genericform, monthlygenericform): firstspecific,
    secondspecific} for a given single input date
    """
    set_of_first_active_spreads = {}
    for tuple in list_of_tuples:
        monthA, monthB, monthly_genericA, monthly_genericB =\
            extract_monthly_gen(product_code, tuple)
        spread_products, AbeforeB = getRelevantCodes(
            monthly_genericA, monthly_genericB, monthB, single_date)
        result = convert_to_specific(
            product_code, spread_products, monthA, monthB, AbeforeB, single_date)
        set_of_first_active_spreads.update(result)
    return set_of_first_active_spreads

print(get_spreads("CO", [('DEC', 'JUN'), ('DEC', 'JUL')], "2025/04/30"))














