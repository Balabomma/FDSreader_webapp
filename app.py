"""
FDSReader — Flask Web Application
====================================
Multi-page Bootstrap 5 interface for fdsreader.
Routes are kept thin; all rendering lives in fds_utils.py.
"""

from __future__ import annotations

import os
import io
import traceback
import zipfile

from flask import (Flask, render_template, request, jsonify,
                   session, send_file)
import fds_utils

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

#  PAGE ROUTES

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/slices")
def slice_viewer():
    return render_template("slice_viewer.html")


@app.route("/boundaries")
def boundary_viewer():
    return render_template("boundary_viewer.html")


@app.route("/devices")
def device_viewer():
    return render_template("device_viewer.html")


@app.route("/hrr")
def hrr_viewer():
    return render_template("hrr_viewer.html")


@app.route("/plot3d")
def plot3d_viewer():
    return render_template("plot3d_viewer.html")


@app.route("/smoke3d")
def smoke3d_viewer():
    return render_template("smoke3d_viewer.html")


@app.route("/particles")
def particle_viewer():
    return render_template("particle_viewer.html")


@app.route("/isosurfaces")
def isosurface_viewer():
    return render_template("isosurface_viewer.html")


@app.route("/evacuation")
def evac_viewer():
    return render_template("evac_viewer.html")


@app.route("/performance")
def performance_viewer():
    return render_template("performance_viewer.html")


#  DIRECTORY BROWSER

@app.route("/api/browse", methods=["GET"])
def api_browse():
    path = request.args.get("path", "")
    try:
        return jsonify(fds_utils.list_directory(path))
    except Exception as e:
        return jsonify({"error": str(e)}), 400


#  LOAD SIMULATION

@app.route("/api/load", methods=["POST"])
def api_load():
    d = request.get_json()
    path = (d.get("path") or "").strip()
    try:
        sim = fds_utils.load_simulation(path)
        session["sim_path"] = path
        # Gather HRR column count
        hrr = getattr(sim, "hrr", None)
        n_hrr = 0
        if hrr is not None:
            try:
                n_hrr = len([k for k in hrr.keys() if k != "Time"])
            except Exception:
                pass

        return jsonify({
            "success": True,
            "info": {
                "chid": getattr(sim, "chid", "N/A"),
                "meshes": len(getattr(sim, "meshes", [])),
                "n_slices": len(getattr(sim, "slices", [])),
                "n_obstructions": len([o for o in (sim.obstructions or [])
                                       if getattr(o, "has_boundary_data", False)]),
                "n_devices": len(getattr(sim, "devices", [])),
                "n_hrr_columns": n_hrr,
                "n_plot3d": len(getattr(sim, "data_3d", []) or []),
                "n_smoke3d": len(getattr(sim, "smoke_3d", []) or []),
                "n_particles": len(getattr(sim, "particles", []) or []),
                "n_isosurfaces": len(getattr(sim, "isosurfaces", []) or []),
                "n_evac": len(getattr(sim, "evacs", []) or []),
                "has_cpu": bool(getattr(sim, "cpu", None)),
                "has_steps": bool(getattr(sim, "steps", None)),
            }
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


# SLICE API

@app.route("/api/slices", methods=["GET"])
def api_slices():
    path = request.args.get("path") or session.get("sim_path")
    if not path:
        return jsonify({"error": "No simulation loaded"}), 400
    try:
        sim = fds_utils.load_simulation(path)
        return jsonify(fds_utils.get_slice_metadata(sim))
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/slice/render", methods=["POST"])
def api_slice_render():
    d = request.get_json()
    path = d.get("path") or session.get("sim_path")
    if not path:
        return jsonify({"error": "No simulation loaded"}), 400
    try:
        sim = fds_utils.load_simulation(path)
        result = fds_utils.render_slice(
            sim,
            slice_id=d.get("slice_id", 0),
            timestep_s=d.get("timestep", 0),
            vmin=d.get("vmin"),
            vmax=d.get("vmax"),
            cmap=d.get("cmap", "jet"),
            show_colorbar=d.get("show_colorbar", True),
            show_labels=d.get("show_labels", True),
            use_global=d.get("use_global", True),
            mesh_index=d.get("mesh_index"),
        )
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/slice/render_multi", methods=["POST"])
def api_slice_render_multi():
    d = request.get_json()
    path = d.get("path") or session.get("sim_path")
    if not path:
        return jsonify({"error": "No simulation loaded"}), 400
    try:
        sim = fds_utils.load_simulation(path)
        result = fds_utils.render_slice_multi(
            sim,
            slice_id=d.get("slice_id", 0),
            timesteps_s=d.get("timesteps", []),
            vmin=d.get("vmin"),
            vmax=d.get("vmax"),
            cmap=d.get("cmap", "jet"),
            use_global=d.get("use_global", True),
            mesh_index=d.get("mesh_index"),
        )
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/slice/animation_frames", methods=["POST"])
def api_slice_animation_frames():
    d = request.get_json()
    path = d.get("path") or session.get("sim_path")
    if not path:
        return jsonify({"error": "No simulation loaded"}), 400
    try:
        sim = fds_utils.load_simulation(path)
        frames = fds_utils.render_slice_animation_frames(
            sim,
            slice_id=d.get("slice_id", 0),
            t_start=d.get("t_start", 0),
            t_end=d.get("t_end"),
            n_frames=d.get("n_frames", 20),
            vmin=d.get("vmin"),
            vmax=d.get("vmax"),
            cmap=d.get("cmap", "jet"),
            use_global=d.get("use_global", True),
            mesh_index=d.get("mesh_index"),
        )
        return jsonify({"frames": frames})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/slice/profile", methods=["POST"])
def api_slice_profile():
    d = request.get_json()
    path = d.get("path") or session.get("sim_path")
    if not path:
        return jsonify({"error": "No simulation loaded"}), 400
    try:
        sim = fds_utils.load_simulation(path)
        result = fds_utils.render_slice_profile(
            sim,
            slice_id=d.get("slice_id", 0),
            direction=d.get("direction", "x"),
            position=d.get("position", 0),
            time_s=d.get("time", 100),
            use_global=d.get("use_global", True),
            mesh_index=d.get("mesh_index"),
        )
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/slice/timeseries", methods=["POST"])
def api_slice_timeseries():
    d = request.get_json()
    path = d.get("path") or session.get("sim_path")
    if not path:
        return jsonify({"error": "No simulation loaded"}), 400
    try:
        sim = fds_utils.load_simulation(path)
        result = fds_utils.render_slice_timeseries(
            sim,
            slice_id=d.get("slice_id", 0),
            point=d.get("point", {}),
            use_global=d.get("use_global", True),
            mesh_index=d.get("mesh_index"),
        )
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# BOUNDARY API

@app.route("/api/obstructions", methods=["GET"])
def api_obstructions():
    path = request.args.get("path") or session.get("sim_path")
    if not path:
        return jsonify({"error": "No simulation loaded"}), 400
    try:
        sim = fds_utils.load_simulation(path)
        return jsonify(fds_utils.get_obstruction_metadata(sim))
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/boundary/render", methods=["POST"])
def api_boundary_render():
    d = request.get_json()
    path = d.get("path") or session.get("sim_path")
    if not path:
        return jsonify({"error": "No simulation loaded"}), 400
    try:
        sim = fds_utils.load_simulation(path)
        result = fds_utils.render_boundary(
            sim,
            obst_id=d.get("obst_id", 0),
            quantity=d.get("quantity", ""),
            orientation=d.get("orientation", 3),
            timestep_s=d.get("timestep", 0),
            vmin=d.get("vmin"),
            vmax=d.get("vmax"),
            cmap=d.get("cmap", "hot"),
            show_colorbar=d.get("show_colorbar", True),
        )
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/boundary/render_multi", methods=["POST"])
def api_boundary_render_multi():
    d = request.get_json()
    path = d.get("path") or session.get("sim_path")
    if not path:
        return jsonify({"error": "No simulation loaded"}), 400
    try:
        sim = fds_utils.load_simulation(path)
        result = fds_utils.render_boundary_multi(
            sim,
            obst_id=d.get("obst_id", 0),
            quantity=d.get("quantity", ""),
            orientation=d.get("orientation", 3),
            timesteps_s=d.get("timesteps", []),
            vmin=d.get("vmin"),
            vmax=d.get("vmax"),
            cmap=d.get("cmap", "hot"),
        )
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/boundary/animation_frames", methods=["POST"])
def api_boundary_animation_frames():
    d = request.get_json()
    path = d.get("path") or session.get("sim_path")
    if not path:
        return jsonify({"error": "No simulation loaded"}), 400
    try:
        sim = fds_utils.load_simulation(path)
        frames = fds_utils.render_boundary_animation_frames(
            sim,
            obst_id=d.get("obst_id", 0),
            quantity=d.get("quantity", ""),
            orientation=d.get("orientation", 3),
            t_start=d.get("t_start", 0),
            t_end=d.get("t_end"),
            n_frames=d.get("n_frames", 20),
            vmin=d.get("vmin"),
            vmax=d.get("vmax"),
            cmap=d.get("cmap", "hot"),
        )
        return jsonify({"frames": frames})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/boundary/timeseries", methods=["POST"])
def api_boundary_timeseries():
    d = request.get_json()
    path = d.get("path") or session.get("sim_path")
    if not path:
        return jsonify({"error": "No simulation loaded"}), 400
    try:
        sim = fds_utils.load_simulation(path)
        result = fds_utils.render_boundary_timeseries(
            sim,
            obst_id=d.get("obst_id", 0),
            quantity=d.get("quantity", ""),
            orientation=d.get("orientation", 3),
        )
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# DEVICE API

@app.route("/api/devices", methods=["GET"])
def api_devices():
    path = request.args.get("path") or session.get("sim_path")
    if not path:
        return jsonify({"error": "No simulation loaded"}), 400
    try:
        sim = fds_utils.load_simulation(path)
        return jsonify(fds_utils.get_device_metadata(sim))
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/device/render", methods=["POST"])
def api_device_render():
    d = request.get_json()
    path = d.get("path") or session.get("sim_path")
    if not path:
        return jsonify({"error": "No simulation loaded"}), 400
    try:
        sim = fds_utils.load_simulation(path)
        result = fds_utils.render_devices(
            sim,
            device_ids=d.get("device_ids", []),
            time_range=d.get("time_range"),
            show_grid=d.get("show_grid", True),
        )
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/device/compare", methods=["POST"])
def api_device_compare():
    d = request.get_json()
    path = d.get("path") or session.get("sim_path")
    if not path:
        return jsonify({"error": "No simulation loaded"}), 400
    try:
        sim = fds_utils.load_simulation(path)
        result = fds_utils.render_device_comparison(
            sim,
            device_ids_left=d.get("device_ids_left", []),
            device_ids_right=d.get("device_ids_right", []),
            time_range=d.get("time_range"),
        )
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


#  HRR API

@app.route("/api/hrr", methods=["GET"])
def api_hrr():
    path = request.args.get("path") or session.get("sim_path")
    if not path:
        return jsonify({"error": "No simulation loaded"}), 400
    try:
        sim = fds_utils.load_simulation(path)
        return jsonify(fds_utils.get_hrr_metadata(sim))
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/hrr/render", methods=["POST"])
def api_hrr_render():
    d = request.get_json()
    path = d.get("path") or session.get("sim_path")
    if not path:
        return jsonify({"error": "No simulation loaded"}), 400
    try:
        sim = fds_utils.load_simulation(path)
        result = fds_utils.render_hrr(
            sim,
            columns=d.get("columns", []),
            time_range=d.get("time_range"),
            show_grid=d.get("show_grid", True),
        )
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


#  PLOT3D API

@app.route("/api/plot3d", methods=["GET"])
def api_plot3d():
    path = request.args.get("path") or session.get("sim_path")
    if not path:
        return jsonify({"error": "No simulation loaded"}), 400
    try:
        sim = fds_utils.load_simulation(path)
        return jsonify(fds_utils.get_plot3d_metadata(sim))
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/plot3d/render", methods=["POST"])
def api_plot3d_render():
    d = request.get_json()
    path = d.get("path") or session.get("sim_path")
    if not path:
        return jsonify({"error": "No simulation loaded"}), 400
    try:
        sim = fds_utils.load_simulation(path)
        result = fds_utils.render_plot3d_cutplane(
            sim,
            p3d_index=d.get("p3d_index", 0),
            time_idx=d.get("time_idx", 0),
            quantity_idx=d.get("quantity_idx", 0),
            axis=d.get("axis", "z"),
            position=d.get("position"),
            vmin=d.get("vmin"),
            vmax=d.get("vmax"),
            cmap=d.get("cmap", "jet"),
            show_colorbar=d.get("show_colorbar", True),
        )
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


#  SMOKE3D API

@app.route("/api/smoke3d", methods=["GET"])
def api_smoke3d():
    path = request.args.get("path") or session.get("sim_path")
    if not path:
        return jsonify({"error": "No simulation loaded"}), 400
    try:
        sim = fds_utils.load_simulation(path)
        return jsonify(fds_utils.get_smoke3d_metadata(sim))
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/smoke3d/render", methods=["POST"])
def api_smoke3d_render():
    d = request.get_json()
    path = d.get("path") or session.get("sim_path")
    if not path:
        return jsonify({"error": "No simulation loaded"}), 400
    try:
        sim = fds_utils.load_simulation(path)
        result = fds_utils.render_smoke3d_cutplane(
            sim,
            smoke_index=d.get("smoke_index", 0),
            time_idx=d.get("time_idx", 0),
            axis=d.get("axis", "z"),
            position=d.get("position"),
            vmin=d.get("vmin"),
            vmax=d.get("vmax"),
            cmap=d.get("cmap", "hot"),
            show_colorbar=d.get("show_colorbar", True),
        )
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


#  PARTICLE API

@app.route("/api/particles", methods=["GET"])
def api_particles():
    path = request.args.get("path") or session.get("sim_path")
    if not path:
        return jsonify({"error": "No simulation loaded"}), 400
    try:
        sim = fds_utils.load_simulation(path)
        return jsonify(fds_utils.get_particle_metadata(sim))
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/particle/scatter", methods=["POST"])
def api_particle_scatter():
    d = request.get_json()
    path = d.get("path") or session.get("sim_path")
    if not path:
        return jsonify({"error": "No simulation loaded"}), 400
    try:
        sim = fds_utils.load_simulation(path)
        result = fds_utils.render_particle_scatter(
            sim,
            class_index=d.get("class_index", 0),
            time_idx=d.get("time_idx", 0),
            plane=d.get("plane", "xy"),
            color_quantity=d.get("color_quantity"),
            cmap=d.get("cmap", "jet"),
            show_colorbar=d.get("show_colorbar", True),
        )
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/particle/histogram", methods=["POST"])
def api_particle_histogram():
    d = request.get_json()
    path = d.get("path") or session.get("sim_path")
    if not path:
        return jsonify({"error": "No simulation loaded"}), 400
    try:
        sim = fds_utils.load_simulation(path)
        result = fds_utils.render_particle_histogram(
            sim,
            class_index=d.get("class_index", 0),
            quantity=d.get("quantity", ""),
            time_idx=d.get("time_idx", 0),
            bins=d.get("bins", 50),
        )
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


#  ISOSURFACE API

@app.route("/api/isosurfaces", methods=["GET"])
def api_isosurfaces():
    path = request.args.get("path") or session.get("sim_path")
    if not path:
        return jsonify({"error": "No simulation loaded"}), 400
    try:
        sim = fds_utils.load_simulation(path)
        return jsonify(fds_utils.get_isosurface_metadata(sim))
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/isosurface/render", methods=["POST"])
def api_isosurface_render():
    d = request.get_json()
    path = d.get("path") or session.get("sim_path")
    if not path:
        return jsonify({"error": "No simulation loaded"}), 400
    try:
        sim = fds_utils.load_simulation(path)
        result = fds_utils.render_isosurface_scatter(
            sim,
            iso_index=d.get("iso_index", 0),
            time_idx=d.get("time_idx", 0),
            plane=d.get("plane", "xy"),
            cmap=d.get("cmap", "hot"),
            show_colorbar=d.get("show_colorbar", True),
        )
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


#  EVACUATION API

@app.route("/api/evacuation", methods=["GET"])
def api_evac():
    path = request.args.get("path") or session.get("sim_path")
    if not path:
        return jsonify({"error": "No simulation loaded"}), 400
    try:
        sim = fds_utils.load_simulation(path)
        return jsonify(fds_utils.get_evac_metadata(sim))
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/evacuation/floorplan", methods=["POST"])
def api_evac_floorplan():
    d = request.get_json()
    path = d.get("path") or session.get("sim_path")
    if not path:
        return jsonify({"error": "No simulation loaded"}), 400
    try:
        sim = fds_utils.load_simulation(path)
        result = fds_utils.render_evac_floorplan(
            sim,
            time_idx=d.get("time_idx", 0),
            class_index=d.get("class_index"),
        )
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/evacuation/timeseries", methods=["POST"])
def api_evac_timeseries():
    d = request.get_json()
    path = d.get("path") or session.get("sim_path")
    if not path:
        return jsonify({"error": "No simulation loaded"}), 400
    try:
        sim = fds_utils.load_simulation(path)
        result = fds_utils.render_evac_timeseries(
            sim,
            metric=d.get("metric", "agents"),
        )
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# PERFORMANCE API (CPU + STEPS)

@app.route("/api/cpu", methods=["GET"])
def api_cpu():
    path = request.args.get("path") or session.get("sim_path")
    if not path:
        return jsonify({"error": "No simulation loaded"}), 400
    try:
        sim = fds_utils.load_simulation(path)
        return jsonify(fds_utils.get_cpu_metadata(sim))
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/cpu/render", methods=["POST"])
def api_cpu_render():
    d = request.get_json()
    path = d.get("path") or session.get("sim_path")
    if not path:
        return jsonify({"error": "No simulation loaded"}), 400
    try:
        sim = fds_utils.load_simulation(path)
        result = fds_utils.render_cpu(
            sim,
            columns=d.get("columns", []),
            time_range=d.get("time_range"),
            show_grid=d.get("show_grid", True),
        )
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/steps", methods=["GET"])
def api_steps():
    path = request.args.get("path") or session.get("sim_path")
    if not path:
        return jsonify({"error": "No simulation loaded"}), 400
    try:
        sim = fds_utils.load_simulation(path)
        return jsonify(fds_utils.get_steps_metadata(sim))
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/steps/render", methods=["POST"])
def api_steps_render():
    d = request.get_json()
    path = d.get("path") or session.get("sim_path")
    if not path:
        return jsonify({"error": "No simulation loaded"}), 400
    try:
        sim = fds_utils.load_simulation(path)
        result = fds_utils.render_steps(
            sim,
            columns=d.get("columns", []),
            time_range=d.get("time_range"),
            show_grid=d.get("show_grid", True),
        )
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


#  DOWNLOAD ENDPOINTS

@app.route("/api/download/slice/gif", methods=["POST"])
def download_slice_gif():
    d = request.get_json()
    path = d.get("path") or session.get("sim_path")
    if not path:
        return jsonify({"error": "No simulation loaded"}), 400
    try:
        sim = fds_utils.load_simulation(path)
        gif_buf, fname = fds_utils.export_slice_gif(
            sim,
            slice_id=d.get("slice_id", 0),
            t_start=d.get("t_start", 0),
            t_end=d.get("t_end"),
            n_frames=d.get("n_frames", 20),
            fps=d.get("fps", 4),
            vmin=d.get("vmin"),
            vmax=d.get("vmax"),
            cmap=d.get("cmap", "jet"),
            use_global=d.get("use_global", True),
            mesh_index=d.get("mesh_index"),
        )
        return send_file(gif_buf, mimetype="image/gif",
                         as_attachment=True, download_name=fname)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/download/boundary/gif", methods=["POST"])
def download_boundary_gif():
    d = request.get_json()
    path = d.get("path") or session.get("sim_path")
    if not path:
        return jsonify({"error": "No simulation loaded"}), 400
    try:
        sim = fds_utils.load_simulation(path)
        gif_buf, fname = fds_utils.export_boundary_gif(
            sim,
            obst_id=d.get("obst_id", 0),
            quantity=d.get("quantity", ""),
            orientation=d.get("orientation", 3),
            t_start=d.get("t_start", 0),
            t_end=d.get("t_end"),
            n_frames=d.get("n_frames", 20),
            fps=d.get("fps", 4),
            vmin=d.get("vmin"),
            vmax=d.get("vmax"),
            cmap=d.get("cmap", "hot"),
        )
        return send_file(gif_buf, mimetype="image/gif",
                         as_attachment=True, download_name=fname)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


#  COLORMAPS

@app.route("/api/colormaps")
def colormaps():
    return jsonify({"colormaps": [
        "jet", "hot", "inferno", "magma", "plasma", "viridis",
        "turbo", "coolwarm", "RdYlBu_r", "Spectral_r", "RdBu",
        "gnuplot2", "YlOrRd", "cividis",
    ]})




if __name__ == "__main__":
    print("=" * 60)
    print("  FDS Viewer — Fire Dynamics Simulation Explorer")
    print("  http://localhost:5000")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=5000)
