import numpy as np
import numpy.typing as npt
import zarr

class RustBatchReader:
    def __new__(cls, zarr_array: zarr.Array) -> RustBatchReader: ...
    def read_ranges(
        self,
        starts: npt.NDArray[np.int64],
        ends: npt.NDArray[np.int64],
    ) -> tuple[npt.NDArray, npt.NDArray[np.int64]]:
        """Read raveled element ranges from a sharded zarr array.

        `starts[i]` and `ends[i]` are raveled element indices in C-order over
        the full N-D array shape. Each range must be last-axis-contiguous
        (stay within a single last-axis row). Returns `(flat_data, lengths)`.
        """
        ...

def bitpack_encode(data: bytes, transform: str) -> npt.NDArray[np.uint8]: ...
def bitpack_decode(
    data: bytes,
) -> npt.NDArray[np.uint8]: ...
