#################################
#
# Cut script example
#
# CPDv21Ge (rough) Chi2 cut
#
#################################
import numpy as np
import vaex
import detanalysis.func as cut


# metadata
@cut.version(1.3)
@cut.authors('A. Person, B. OtherPerson')
@cut.contact('a.person@gmail.com (A. Person)')
@cut.description('A low chi2 cut for demo')
@cut.date('2/21/2023')


def cut_chi2_CPDv21Ge(df):
    """
    Preliminary OF chi2 cut 

    comment here

    """

    # 2nd order polynomial
    coeff = [13.5, -3.3, 0.4]
    poly = np.polyval(coeff, df.amp_of1x1_constrained_CPDv21Ge*1e5)*1e6

    
    # polynomial cut below amp=1e-5
    cut =  ((df.lowchi2_of1x1_constrained_CPDv21Ge<poly)
            & (df.amp_of1x1_constrained_CPDv21Ge<1e-5))
    
    # keep everything above it
    cut = cut | (df.amp_of1x1_constrained_CPDv21Ge>=1e-5)

    
    return cut
           
