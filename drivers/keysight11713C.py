class Keysight11713C:
    rm = pyvisa.ResourceManager()
    swc = rm.open_resource("swc1")

    def __init__(self, bank_id):

        """
        Parameters:
            bank_id: str
                "X" or "Y"
        """
        self.bank_id = bank_id
        if (not (bank_id == "X" or bank_id == "Y")):
            print("Such kind of bank does not exist")

    def set_attenuation(self, attenuation):
        if (attenuation > 81):
            raise ValueError(f"Attenuation value {attenuation} is too high")
        elif attenuation < 0:
            raise ValueError(f"Attenuation value {attenuation} can't be negative")

        bank2_value = (attenuation // 10) * 10
        bank1_value = (attenuation % 10)
        if (attenuation >= 70):
            bank2_value = 70
            bank1_value = attenuation - bank2_value
        swc.write("ATTenuator:BANK1:" + self.bank_id + " " + str(bank1_value))
        swc.write("ATTenuator:BANK2:" + self.bank_id + " " + str(bank2_value))

    def get_attenuation(self):
        attenuation = int(swc.query("ATTenuator:BANK2:" + str(self.bank_id) + "?")) + int(
            swc.query("ATTenuator:BANK1:" + str(self.bank_id) + "?"))
        return (attenuation)

