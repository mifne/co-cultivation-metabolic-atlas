"""Numeric-only current LP batches for fixed-layout GPU transformations.

CSR patterns and dimensions belong to a separately validated layout. No
acceptance, hash or historical LP solution is carried by these arrays.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceLPBatch:
    data: object  # Flat concatenated canonical CSR values, lane order.
    rhs: object   # [batch, rows]
    lower: object # [batch, columns]
    upper: object # [batch, columns]
    c: object     # [batch, columns]

    def arrays(self):
        return (self.data, self.rhs, self.lower, self.upper, self.c)
