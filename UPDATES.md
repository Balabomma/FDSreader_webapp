# Plot3D & Smoke3D Multi-Timestep Feature Updates

## Overview
This update adds comprehensive multi-timestep rendering capabilities to Plot3D and Smoke3D viewers, matching the features already available in the Slice and Boundary viewers.

## New Features

### Backend (fds_utils.py)
1. **Plot3D Multi-Timestep Functions:**
   - `render_plot3d_multi()` - Render multi-timestep subplot grid with shared color scale
   - `render_plot3d_animation_frames()` - Generate animation frames for playback

2. **Smoke3D Multi-Timestep Functions:**
   - `render_smoke3d_multi()` - Render multi-timestep subplot grid with shared color scale
   - `render_smoke3d_animation_frames()` - Generate animation frames for playback

### API Endpoints (app.py)
1. `/api/plot3d/render_multi` - POST endpoint for multi-timestep plot3d rendering
2. `/api/plot3d/animation_frames` - POST endpoint for plot3d animation frame generation
3. `/api/smoke3d/render_multi` - POST endpoint for multi-timestep smoke3d rendering
4. `/api/smoke3d/animation_frames` - POST endpoint for smoke3d animation frame generation

### Frontend UI Enhancements

#### Plot3D Viewer (plot3d_viewer.html)
- Added View Mode selector: Single Timestep / Multi-Timestep Grid / Animation
- Multi-select dropdown for timestep selection (with Ctrl+Click support)
- Animation controls with frame count input
- Select All / Clear All buttons for timesteps

#### Smoke3D Viewer (smoke3d_viewer.html)
- Added View Mode selector: Single Timestep / Multi-Timestep Grid / Animation
- Multi-select dropdown for timestep index selection (with Ctrl+Click support)
- Animation controls with frame count input
- Select All / Clear All buttons for timesteps

### JavaScript Functions (viewer.js)

#### Plot3D:
- `onP3dViewModeChange()` - Toggle between view modes
- `p3dSelectAllTimes()` - Select all timesteps for multi-view
- `p3dClearTimes()` - Clear timestep selection
- `renderPlot3D()` - Enhanced to support all three render modes
- `_showP3dAnimFrame()` - Display animation frames with playback

#### Smoke3D:
- `onS3dViewModeChange()` - Toggle between view modes
- `s3dSelectAllTimes()` - Select all timesteps for multi-view
- `s3dClearTimes()` - Clear timestep selection
- `renderSmoke3D()` - Enhanced to support all three render modes
- `_showS3dAnimFrame()` - Display animation frames with playback

## How to Use

### Multi-Timestep Grid
1. Select a Plot3D or Smoke3D dataset
2. Change "View Mode" to "Multi-Timestep Grid"
3. Select 2 or more timesteps using the multi-select dropdown
4. Click "Render"
5. View the subplot grid with shared color scale

### Animation
1. Select a Plot3D or Smoke3D dataset
2. Change "View Mode" to "Animation"
3. Set animation parameters (t_start, t_end, n_frames)
4. Click "Render"
5. Use play/pause controls to animate through frames

### Single Timestep (Original Behavior)
- Select "Single Timestep" view mode
- Choose a single timestep
- Click "Render"

## Technical Details

### Shared Color Scale
Both multi-timestep renderings compute a global min/max across all selected timesteps, ensuring consistent color mapping across the grid.

### Animation Frames
The animation system generates frames at specified time intervals with automatic timestep interpolation.

### Plot Styling
- All visualizations use the existing dark theme with consistent formatting
- Multi-timestep grids use 2-column layout with proper spacing
- Shared colorbars show quantity and units

## Files Modified
- `fds_utils.py` - Added 4 new rendering functions
- `app.py` - Added 4 new API endpoints
- `templates/plot3d_viewer.html` - Updated UI with view mode selector
- `templates/smoke3d_viewer.html` - Updated UI with view mode selector
- `static/js/viewer.js` - Added 12 new JavaScript functions

## Backward Compatibility
All changes are backward compatible. Single-timestep rendering works exactly as before, with the new features accessible through the new "View Mode" dropdown.

