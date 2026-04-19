#################################
#
# Derived feature script example
#
# CPDv21Ge simple slope
#
#################################

import vaex
import detanalysis.func as feature


# metadata
@feature.version(1.3)
@feature.authors('A. Person')
@feature.contact('a.person@gmail.com (A. Person)')
@feature.description('Simple slope CPDv21Ge')
@feature.date('2/21/2023')


def slope_CPDv21Ge(df):
    """
    Slope CDPSv21Ge

    comment here

    """

    slope = (df.baseline_end_CPDv21Ge
             - df.baseline_start_CPDv21Ge)
    
    return slope

