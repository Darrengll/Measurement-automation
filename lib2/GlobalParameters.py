

class GlobalParameters():

    resonator_types = {"reflection": False, "transmission": False}

    which_sweet_spot = {"I": "top", "II":"top", "III":"top",
                        "IV":"top", "VI":"top", "V":"top",
                        "VII":"top", "VIII":"top"}


    recalibrate_mixers = {"I": True, "II":True, "III":True,
                        "IV":True, "V":True, "VI":True,
                        "VII":False, "VIII":False}


    ro_ssb_power = {"I": -50, "II":-50, "III":-30,
                        "IV":-50, "VI":-50, "V":-40,
                        "VII":-50, "VIII":-50}

    exc_ssb_power = {"I": -20, "II":-20, "III":-20,
                        "IV":-20, "VI":-20, "V":-20,
                        "VII":-20, "VIII":-20}
