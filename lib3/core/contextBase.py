import numpy as np
import json


class ContextBase:
    def __init__(self):
        self._equipment = {}
        self._comment = ""

    def get_equipment(self):
        return self._equipment

    def to_string(self):
        self._equipment.update({"comment:": self._comment})

        import json
        import datetime

        class Encoder(json.JSONEncoder):
            def default(self, obj):
                if hasattr(obj, "toJSON"):
                    return obj.toJSON()
                if isinstance(obj, np.ndarray) or \
                        isinstance(obj, datetime.datetime) or \
                        isinstance(obj, np.int32):
                    return obj.__str__()
                else:
                    return json.JSONEncoder.default(self, obj)

        def nice_dict(d):
            return json.dumps(d, indent=4, cls=Encoder)

        return str(nice_dict(self._equipment))

    def update(self, equipment={}, comment=""):
        self._equipment.update(equipment)
        self._comment.join(comment)