import pyvisa


class Keysight11713C:

    def __init__(self, alias, bank_id):

        """
        Parameters:
            alias: str
                name of your device
            bank_id: str
                "X" or "Y"
        """
        rm = pyvisa.ResourceManager()
        self.swc = rm.open_resource(alias)
        self.bank_id = bank_id
        if not (bank_id == "X" or bank_id == "Y"):
            raise ValueError(f"Bank id {bank_id} is not allowed! Use 'X' or 'Y'")

    def set_attenuation(self, attenuation):
        if attenuation > 81:
            raise ValueError(f"Attenuation value {attenuation} is too high")
        elif attenuation < 0:
            raise ValueError(f"Attenuation value {attenuation} can't be negative")

        bank2_value = (attenuation // 10) * 10
        bank1_value = (attenuation % 10)
        if attenuation >= 70:
            bank2_value = 70
            bank1_value = attenuation - bank2_value
        self.swc.write("ATTenuator:BANK1:" + self.bank_id + " " + str(bank1_value))
        self.swc.write("ATTenuator:BANK2:" + self.bank_id + " " + str(bank2_value))

    def get_attenuation(self):
        attenuation = int(self.swc.query("ATTenuator:BANK2:" + str(self.bank_id) + "?")) + int(
            self.swc.query("ATTenuator:BANK1:" + str(self.bank_id) + "?"))
        return attenuation





