

class GlobalParameters:

    resonator_types = {"reflection": False, "transmission": True}

    which_sweet_spot = {"I": "top", "II":"top", "III":"top",
                        "IV":"top", "VI":"top", "V":"top",
                        "VII":"top", "VIII":"top"}


    recalibrate_mixers = {"I": False, "II":False, "III":False,
                        "IV":False, "V":False, "VI":False,
                        "VII":False, "VIII":False}


    ro_ssb_power = {"I": -55, "II":-55, "III":-60,
                        "IV":-60, "VI":-55, "V":-60,
                        "VII":-60, "VIII":-60}

    exc_ssb_power = {"I": -20, "II":-20, "III":-20,
                        "IV":-20, "VI":-20, "V":-20,
                        "VII":-20, "VIII":-20}

    spectroscopy_readout_power = -60

    spectroscopy_excitation_power = 0

    anticrossing_oracle_hits = ["fqmax_below"]
