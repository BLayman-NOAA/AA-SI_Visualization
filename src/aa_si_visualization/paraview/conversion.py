"""
ParaView VTK conversion utilities for sonar/echogram xarray datasets.
"""

import numpy as np
import pyvista as pv
from pathlib import Path


def sv_dataset_to_vtk(ds_Sv, output_path, channel_index_for_bounds=0):
    """
    Convert an xarray Sv dataset to a PyVista RectilinearGrid VTK file
    with all frequency channels as separate cell data arrays.

    Args:
        ds_Sv: xarray.Dataset containing 'Sv', 'ping_time', 'range_sample',
               'echo_range', and 'frequency_nominal' variables.
        output_path: Path (str or Path) for the output .vtk file.
        channel_index_for_bounds: Channel index used to determine the valid
                                  data bounding box (default: 0).

    Returns:
        pv.RectilinearGrid: The grid object that was saved.
    """
    output_path = Path(output_path)

    sv_data_all_channels = ds_Sv["Sv"].values  # (channel, ping_time, range_sample)
    ping_times = ds_Sv["ping_time"].values
    echo_range = ds_Sv["echo_range"].isel(channel=channel_index_for_bounds).values
    frequencies = ds_Sv["frequency_nominal"].values

    # Convert ping_time to seconds from start
    time_numeric = (ping_times - ping_times[0]).astype("timedelta64[s]").astype(float)

    # Find valid-data bounding box from the reference channel
    sv_ref = sv_data_all_channels[channel_index_for_bounds]
    valid_mask = ~np.isnan(sv_ref)
    valid_ping_indices, valid_range_indices = np.where(valid_mask)

    ping_min_idx = valid_ping_indices.min()
    ping_max_idx = valid_ping_indices.max()
    range_min_idx = valid_range_indices.min()
    range_max_idx = valid_range_indices.max()

    # Crop all channels to bounding box
    sv_cropped = sv_data_all_channels[
        :, ping_min_idx : ping_max_idx + 1, range_min_idx : range_max_idx + 1
    ]
    time_cropped = time_numeric[ping_min_idx : ping_max_idx + 1]
    echo_range_cropped = echo_range[
        ping_min_idx : ping_max_idx + 1, range_min_idx : range_max_idx + 1
    ]
    depth_coords = echo_range_cropped[0, :]

    # Build coordinate arrays (one extra node per axis for cell-centred data)
    x_coords = np.concatenate(
        [time_cropped, [time_cropped[-1] + (time_cropped[-1] - time_cropped[-2])]]
    )
    y_coords = np.concatenate(
        [-depth_coords, [-depth_coords[-1] - (depth_coords[-1] - depth_coords[-2])]]
    )
    z_coords = np.array([0])

    grid = pv.RectilinearGrid(x_coords, y_coords, z_coords)

    # Add each channel as a cell data array
    for ch_idx in range(sv_data_all_channels.shape[0]):
        sv_flat = sv_cropped[ch_idx].T.flatten()

        # Pad if grid has more cells than data points
        if sv_flat.size < grid.n_cells:
            sv_flat = np.concatenate(
                [sv_flat, np.full(grid.n_cells - sv_flat.size, np.nan)]
            )

        freq_khz = int(frequencies[ch_idx] / 1000)
        grid.cell_data[f"Sv_dB_Ch{ch_idx}_{freq_khz}kHz"] = sv_flat

    # Metadata
    grid.field_data["n_channels"] = np.array([sv_data_all_channels.shape[0]])
    grid.field_data["frequencies_hz"] = frequencies
    grid.field_data["time_range_seconds"] = np.array([x_coords[0], x_coords[-1]])
    grid.field_data["depth_range_meters"] = np.array([y_coords[0], y_coords[-1]])
    grid.field_data["aspect_ratio_m_per_s"] = np.array(
        [(y_coords[-1] - y_coords[0]) / (x_coords[-1] - x_coords[0])]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(str(output_path), binary=True)

    print(f"Saved {sv_data_all_channels.shape[0]} channels to: {output_path}")
    print(f"Channel names: {list(grid.cell_data.keys())}")

    return grid