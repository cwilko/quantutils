import py_vollib.black_scholes_merton.implied_volatility as iv
import py_vollib_vectorized
import traceback


def get_IV(optionData, rfRate=0.0):

    try:
        impl = iv.implied_volatility(optionData["bid"], optionData["underlying"], optionData["strike"], optionData["timeToExpiry"] / 365., rfRate, optionData["type"], q=0)
    except Exception:
        traceback.print_exc()
    pass

    # Set any below-intrinsic IVs to 0 IV
    #impl["IV"][impl["IV"].isnull()] = 0

    return impl["IV"]
