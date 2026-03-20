import qontrol
import numpy as np

class VoltageView:
    def __init__(self, parent):
        self._p = parent

    def __getitem__(self, key):
        return self._p._link.v[key]

    def __setitem__(self, key, value):

        if isinstance(value, (list, tuple, np.ndarray)):
            for v, k in zip(value, key):
                self._check(v)
                self._check_front(v, k)
        else:
            self._check(value)
            self._check_front(value, key)

        self._p._link.v[key] = value

    def _check(self, value):
        if not isinstance(value, (int, float)):
            raise ValueError("Voltage must be a number")

        if value < self._p._min_voltage or value > self._p._max:
            raise ValueError(
                f"Voltage must be between {self._p._min_voltage} and {self._p._max}"
            )

    def _check_front(self, value, key):
        if not isinstance(value, (int, float)):
            raise ValueError("Voltage must be a number")
        
        tmp = self._p._link.v
        tmp[key] = value
        for i in range(tmp//2):
            if (tmp[i]>self._p._max and tmp[2*i]>0) or (tmp[2*i]>self._p._max and tmp[i]>0):
                raise ValueError(f"You can excide the maximum voltage of {self._p._max_voltage}V only if the channel in front is set to 0V.")
            

    def __len__(self):
        return self._p._n_chs

    def __repr__(self):
        return repr(np.array(self._p._link.v))

    def __array__(self):
        return np.array(self._p._link.v)
    
class CurrentView:
    def __init__(self, parent):
        self._p = parent

    def __getitem__(self, key):
        return self._p._link.i[key]

    def __setitem__(self, key, value):

        if isinstance(value, (list, tuple, np.ndarray)):
            for i, k in zip(value, key):
                self._check(i)
                self._check_front(i, k)
        else:
            self._check(value)
            self._check_front(value, key)

        self._p._link.i[key] = value

    def _check(self, value):
        if not isinstance(value, (int, float)):
            raise ValueError("Current must be a number")

        if value < self._p._min_current or value > self._p._max_current:
            raise ValueError(
                f"Current must be between {self._p._min_current} and {self._p._max_current}"
            )

    def _check_front(self, value, key):
        if not isinstance(value, (int, float)):
            raise ValueError("Current must be a number")
        
        tmp = self._p._link.i
        tmp[key] = value
        for i in range(tmp//2):
            if (tmp[i]>self._p._max_current and tmp[2*i]>0) or (tmp[2*i]>self._p._max_current and tmp[i]>0):
                raise ValueError(f"You can excide the maximum current of {self._p._max_current}A only if the channel in front is set to 0V.")
            

    def __len__(self):
        return self._p._n_chs

    def __repr__(self):
        return repr(np.array(self._p._link.i))

    def __array__(self):
        return np.array(self._p._link.i)

class QontrolPowerSupply():
    def __init__(self, address):
        self._link = qontrol.QXOutput(serial_port_name = address)
        self._n_chs = self._link.n_chs
        self._min_voltage = 0
        self._max_voltage = 5
        self._max = 8
        self._link.vmax= 8
        if self._max_voltage > 5:
            print("Warning: The maximum voltage is set above 5V. Please ensure that your hardware can handle this voltage to avoid damage.")
        if self._max_voltage > self._max:
            print(f"Warning: The maximum voltage is set above the recommended limit of {self._max}V. The hardware cannot handle this voltage. We set the max voltage to 5V to prevent damage.")
            self._max_voltage = 5
        self._voltage = VoltageView(self)
        self._current = CurrentView(self)

    def __del__(self):
        self._link.close()

    def close(self):
        self._link.close()
    
    @property
    def voltage(self):
        return self._voltage
    
    @voltage.setter
    def voltage(self, values):
        if not isinstance(values, (list, tuple, np.ndarray)):
            raise ValueError("Voltage must be a list or tuple.")
        else:
            for value in values:
                if not isinstance(value, (int, float)):
                    raise ValueError("Each voltage value must be a number.")
                if value < self._min_voltage or value > self._max_voltage:
                    raise ValueError(f"Voltage values must be between {self._min_voltage} and {self._max_voltage}.\n To set a voltage up to 8V change just the single channel voltage using the voltage property, e.g. power_supply.voltage[0] = 6")
        if len(values) != self._n_chs:
            raise ValueError(f"Voltage must be a list of {self._n_chs} values.")
        
        self._link.v = values
        print(f"Voltage set to: {values}")

    
    @property
    def current(self):
        return self._current