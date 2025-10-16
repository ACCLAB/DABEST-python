import pytest
import numpy as np
import matplotlib.pyplot as plt
from dabest.multi import MultiContrast, combine
from data.mocked_data_test_multi import (
    two_group_contrast_1,
    two_group_contrast_2,
    two_group_contrast_3,
    delta2_contrast_1,
    delta2_contrast_2,
    minimeta_contrast_1,
    minimeta_contrast_2
)


def test_multicontrast_init_basic():
    """Test basic MultiContrast initialization."""
    mc = MultiContrast(
        dabest_objs=[two_group_contrast_1, two_group_contrast_2]
    )
    
    assert mc is not None
    assert isinstance(mc.structure, dict)
    assert mc.effect_size == "mean_diff"
    assert mc.ci_type == "bca"


def test_multicontrast_init_custom_params():
    """Test MultiContrast initialization with custom parameters."""
    mc = MultiContrast(
        dabest_objs=[two_group_contrast_1, two_group_contrast_2],
        labels=["Treatment A", "Treatment B"],
        effect_size="hedges_g",
        ci_type="pct"
    )
    
    assert mc.effect_size == "hedges_g"
    assert mc.ci_type == "pct"
    assert mc.structure['col_labels'] == ["Treatment A", "Treatment B"]



def test_multicontrast_bootstraps_property():
    """Test that bootstraps property returns data correctly."""
    mc = MultiContrast(
        dabest_objs=[two_group_contrast_1, two_group_contrast_2]
    )
    
    bootstraps = mc.bootstraps
    
    assert isinstance(bootstraps, list)
    assert len(bootstraps) > 0
    # Each bootstrap should be an array-like
    for bs in bootstraps:
        assert hasattr(bs, '__len__')


def test_multicontrast_effect_sizes_property():
    """Test that effect_sizes property returns data correctly."""
    mc = MultiContrast(
        dabest_objs=[two_group_contrast_1, two_group_contrast_2]
    )
    
    effect_sizes = mc.effect_sizes
    
    assert isinstance(effect_sizes, list)
    assert len(effect_sizes) == 2  # Two contrasts


def test_multicontrast_ci_lows_property():
    """Test that ci_lows property returns data correctly."""
    mc = MultiContrast(
        dabest_objs=[two_group_contrast_1, two_group_contrast_2]
    )
    
    ci_lows = mc.ci_lows
    
    assert isinstance(ci_lows, list)
    assert len(ci_lows) == 2


def test_multicontrast_ci_highs_property():
    """Test that ci_highs property returns data correctly."""
    mc = MultiContrast(
        dabest_objs=[two_group_contrast_1, two_group_contrast_2]
    )
    
    ci_highs = mc.ci_highs
    
    assert isinstance(ci_highs, list)
    assert len(ci_highs) == 2


def test_multicontrast_get_bootstrap_by_position_1d():
    """Test getting bootstrap data by position for 1D structure."""
    mc = MultiContrast(
        dabest_objs=[two_group_contrast_1, two_group_contrast_2]
    )
    
    # In 1D, all contrasts are in row 0
    bootstrap = mc.get_bootstrap_by_position(0, 0)
    
    assert bootstrap is not None
    assert hasattr(bootstrap, '__len__')


def test_multicontrast_get_bootstrap_by_position_2d():
    """Test getting bootstrap data by position for 2D structure."""
    mc = MultiContrast(
        dabest_objs=[[two_group_contrast_1, two_group_contrast_2],
                     [two_group_contrast_3, two_group_contrast_1]]
    )
    
    bootstrap = mc.get_bootstrap_by_position(1, 1)
    
    assert bootstrap is not None
    assert hasattr(bootstrap, '__len__')


def test_multicontrast_get_bootstrap_by_position_out_of_bounds():
    """Test that out-of-bounds position raises IndexError."""
    mc = MultiContrast(
        dabest_objs=[two_group_contrast_1, two_group_contrast_2]
    )
    
    error_msg = "out of bounds"
    with pytest.raises(IndexError) as excinfo:
        mc.get_bootstrap_by_position(10, 10)
    
    assert error_msg in str(excinfo.value)


def test_multicontrast_get_bootstrap_mixed_types():
    """Test getting bootstrap data from mixed-type MultiContrast."""
    mc = combine(
        dabest_objs=[[two_group_contrast_1, two_group_contrast_2],
                     [delta2_contrast_1, delta2_contrast_2]],
        allow_mixed_types=True
    )
    
    # Get bootstrap from standard contrast row
    bootstrap_two_group = mc.get_bootstrap_by_position(0, 0)
    # Get bootstrap from delta2 contrast row
    bootstrap_delta2 = mc.get_bootstrap_by_position(1, 0)
    
    assert bootstrap_two_group is not None
    assert bootstrap_delta2 is not None



def test_multicontrast_repr_mixed():
    """Test __repr__ for mixed contrast types."""
    mc = combine(
        dabest_objs=[[two_group_contrast_1, two_group_contrast_2],
                     [delta2_contrast_1, delta2_contrast_2]],
        allow_mixed_types=True
    )
    
    repr_str = repr(mc)
    
    assert "MultiContrast" in repr_str
    assert "mixed" in repr_str


def test_validate_individual_dabest_obj_missing_attribute():
    """Test that validation catches missing required attributes."""
    # Create a mock object without required attributes
    class FakeDabest:
        pass
    
    fake_obj = FakeDabest()
    
    with pytest.raises(AttributeError):
        mc = MultiContrast(dabest_objs=[fake_obj])


def test_validate_effect_size_compatibility_delta2():
    """Test effect size validation for delta2 contrasts."""
    error_msg = "delta-delta analyses only support mean_diff, hedges_g, and delta_g"
    
    with pytest.raises(ValueError) as excinfo:
        MultiContrast(
            dabest_objs=[delta2_contrast_1],
            effect_size="cohens_d"
        )
    
    assert error_msg in str(excinfo.value)


def test_validate_effect_size_compatibility_minimeta():
    """Test effect size validation for mini-meta contrasts."""
    error_msg = "mini-meta analyses only support mean_diff"
    
    with pytest.raises(ValueError) as excinfo:
        MultiContrast(
            dabest_objs=[minimeta_contrast_1],
            effect_size="hedges_g"
        )
    
    assert error_msg in str(excinfo.value)

def test_multicontrast_structure_1d():
    """Test structure property for 1D dabest_objs."""
    mc = MultiContrast(
        dabest_objs=[two_group_contrast_1, two_group_contrast_2, two_group_contrast_3]
    )
    
    assert mc.structure['type'] == '1D'
    assert mc.structure['n_rows'] == 1
    assert mc.structure['n_cols'] == 3
    assert mc.structure['total_dabest_objs'] == 3


def test_multicontrast_structure_2d():
    """Test structure property for 2D dabest_objs."""
    mc = MultiContrast(
        dabest_objs=[[two_group_contrast_1, two_group_contrast_2],
                     [two_group_contrast_3, two_group_contrast_1]]
    )
    
    assert mc.structure['type'] == '2D'
    assert mc.structure['n_rows'] == 2
    assert mc.structure['n_cols'] == 2
    assert mc.structure['total_dabest_objs'] == 4


def test_multicontrast_structure_1d_to_2d():
    """Test that structure normalizes 1D to 2D internally."""
    mc = MultiContrast(
        dabest_objs=[two_group_contrast_1, two_group_contrast_2]
    )
    
    # Even for 1D input, internally stored as 2D
    assert len(mc.structure['dabest_objs_2d']) == 1
    assert len(mc.structure['dabest_objs_2d'][0]) == 2


def test_multicontrast_contrast_type_homogeneous_standard():
    """Test contrast_type for homogeneous standard contrasts."""
    mc = MultiContrast(
        dabest_objs=[two_group_contrast_1, two_group_contrast_2]
    )
    
    assert mc.contrast_type == 'delta'


def test_multicontrast_contrast_type_homogeneous_delta2():
    """Test contrast_type for homogeneous delta2 contrasts."""
    mc = MultiContrast(
        dabest_objs=[delta2_contrast_1, delta2_contrast_2]
    )
    
    assert mc.contrast_type == 'delta2'


def test_multicontrast_contrast_type_mixed():
    """Test contrast_type for mixed contrasts."""
    mc = combine(
        dabest_objs=[[two_group_contrast_1, two_group_contrast_2],
                     [delta2_contrast_1, delta2_contrast_2]],
        allow_mixed_types=True
    )
    
    assert isinstance(mc.contrast_type, dict)
    assert mc.contrast_type['mixed'] == True
    assert 'delta' in mc.contrast_type['unique_types']
    assert 'delta2' in mc.contrast_type['unique_types']


def test_multicontrast_contrast_type_by_row():
    """Test that mixed contrast types track row-wise types."""
    mc = combine(
        dabest_objs=[[two_group_contrast_1, two_group_contrast_2],
                     [delta2_contrast_1, delta2_contrast_2]],
        allow_mixed_types=True
    )
    
    assert 'by_row' in mc.contrast_type
    assert len(mc.contrast_type['by_row']) == 2
    assert mc.contrast_type['by_row'][0] == 'delta'
    assert mc.contrast_type['by_row'][1] == 'delta2'