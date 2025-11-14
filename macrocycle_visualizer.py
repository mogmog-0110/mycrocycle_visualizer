#!/usr/bin/env python3
"""
Macrocycle Formation 3D Visualization Tool
==========================================

3D molecular structure visualization tool for macrocycle formation simulations.
Supports spherical molecules (Si-containing compounds) and cylindrical linkers.

Features:
- Multiple viewpoint rendering
- Depth-based transparency and color shading
- Zoom views for detailed structural analysis
- 4-panel multi-view figures
- Rotation animation (GIF)
- Automatic statistical analysis

Input Format:
    Line 1: num_spheres num_cylinders num_links
    Lines 2 to N+1: sphere coordinates (x y z)
    Lines N+2 to N+M+1: cylinder coordinates (x1 y1 z1 x2 y2 z2)
    Remaining lines: link data (sphere_index cylinder_index)

Usage:
    python macrocycle_visualizer.py --input data.txt --all
    python macrocycle_visualizer.py --input data.txt --multi-view
    python macrocycle_visualizer.py --input data.txt --zoom-only

Author: Shiggy
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.patches import Patch
from collections import defaultdict
import argparse
import os
import sys
from typing import List, Tuple, Dict, Optional


class SimulationData:
    """Manages molecular structure data from simulation output files."""
    
    def __init__(self, filename: str):
        """
        Load simulation data from file.
        
        Args:
            filename: Path to input file
        """
        self.filename = filename
        self.spheres = None
        self.cylinders = None
        self.links = None
        self.num_spheres = 0
        self.num_cylinders = 0
        self.load_data()
    
    def load_data(self):
        """Parse input file and extract structure data."""
        with open(self.filename, 'r') as f:
            lines = f.readlines()
        
        # Parse header: num_spheres num_cylinders num_links
        first_line = lines[0].strip().split()
        self.num_spheres = int(first_line[0])
        self.num_cylinders = int(first_line[1])
        
        # Parse sphere coordinates
        spheres = []
        for i in range(1, self.num_spheres + 1):
            coords = list(map(float, lines[i].strip().split()))
            spheres.append(coords[:3])
        self.spheres = np.array(spheres)
        
        # Parse cylinder coordinates (start and end points)
        cylinders = []
        cylinder_start_line = self.num_spheres + 1
        for i in range(cylinder_start_line, cylinder_start_line + self.num_cylinders):
            coords = list(map(float, lines[i].strip().split()))
            p1 = coords[:3]
            p2 = coords[3:6]
            cylinders.append((p1, p2))
        self.cylinders = cylinders
        
        # Parse connectivity links between spheres and cylinders
        links = []
        link_start_line = self.num_spheres + self.num_cylinders + 1
        for i in range(link_start_line, len(lines)):
            link_data = list(map(int, lines[i].strip().split()))
            if len(link_data) >= 2:
                links.append((link_data[0], link_data[1]))
        self.links = links
    
    def get_statistics(self) -> Dict:
        """
        Calculate structural statistics.
        
        Returns:
            Dictionary containing counts, ranges, and connectivity metrics
        """
        stats = {
            'num_spheres': self.num_spheres,
            'num_cylinders': self.num_cylinders,
            'num_links': len(self.links),
            'avg_links_per_sphere': len(self.links) / self.num_spheres if self.num_spheres > 0 else 0,
        }
        
        # Calculate spatial extents
        stats['x_range'] = float(self.spheres[:, 0].max() - self.spheres[:, 0].min())
        stats['y_range'] = float(self.spheres[:, 1].max() - self.spheres[:, 1].min())
        stats['z_range'] = float(self.spheres[:, 2].max() - self.spheres[:, 2].min())
        
        # Analyze connectivity distribution
        sphere_link_count = defaultdict(int)
        for sphere_idx, _ in self.links:
            sphere_link_count[sphere_idx] += 1
        
        if sphere_link_count:
            link_counts = list(sphere_link_count.values())
            stats['min_links'] = min(link_counts)
            stats['max_links'] = max(link_counts)
            stats['avg_links'] = float(np.mean(link_counts))
        
        return stats
    
    def print_statistics(self):
        """Display structural statistics to console."""
        stats = self.get_statistics()
        print("\n" + "="*70)
        print("STRUCTURE STATISTICS")
        print("="*70)
        print(f"Total Spheres (Si-molecules):        {stats['num_spheres']}")
        print(f"Total Cylinders (Linkers):           {stats['num_cylinders']}")
        print(f"Total Links (Si-O bonds):            {stats['num_links']}")
        print(f"Average links per sphere:            {stats['avg_links_per_sphere']:.2f}")
        
        print(f"\nSpatial dimensions:")
        print(f"  X: {stats['x_range']:.2f} Å")
        print(f"  Y: {stats['y_range']:.2f} Å")
        print(f"  Z: {stats['z_range']:.2f} Å")
        
        if 'min_links' in stats:
            print(f"\nConnectivity distribution:")
            print(f"  Min links per sphere: {stats['min_links']}")
            print(f"  Max links per sphere: {stats['max_links']}")
            print(f"  Avg links per sphere: {stats['avg_links']:.2f}")
        
        print("="*70 + "\n")


class DepthShading:
    """Calculates depth-based colors and transparency for enhanced 3D perception."""
    
    @staticmethod
    def calculate_depths(positions: np.ndarray, elev: float = 30, azim: float = 45) -> np.ndarray:
        """
        Calculate normalized depth values from viewpoint.
        
        Args:
            positions: 3D coordinates array (N, 3)
            elev: Elevation angle of viewpoint
            azim: Azimuth angle of viewpoint
            
        Returns:
            Normalized depth values (0: near, 1: far)
        """
        center = np.mean(positions, axis=0)
        distances = np.linalg.norm(positions - center, axis=1)
        max_dist = np.max(distances)
        
        if max_dist > 0:
            return distances / max_dist
        return np.zeros(len(positions))
    
    @staticmethod
    def get_alphas(depths: np.ndarray, 
                   alpha_min: float = 0.3, 
                   alpha_max: float = 0.9) -> np.ndarray:
        """
        Convert depth values to transparency levels.
        
        Args:
            depths: Normalized depth values
            alpha_min: Minimum alpha (far objects)
            alpha_max: Maximum alpha (near objects)
            
        Returns:
            Alpha values array
        """
        return alpha_min + (alpha_max - alpha_min) * (1 - depths)
    
    @staticmethod
    def get_colors(depths: np.ndarray, 
                   colormap_name: str = 'Reds',
                   intensity_min: float = 0.4,
                   intensity_max: float = 1.0) -> np.ndarray:
        """
        Convert depth values to color intensities.
        
        Args:
            depths: Normalized depth values
            colormap_name: Matplotlib colormap name
            intensity_min: Minimum color intensity (far objects)
            intensity_max: Maximum color intensity (near objects)
            
        Returns:
            RGBA color array
        """
        cmap = plt.colormaps.get_cmap(colormap_name)
        intensities = intensity_min + (intensity_max - intensity_min) * (1 - depths)
        return cmap(intensities)


class StructureDetector:
    """Detects structural features such as cyclic tetramers."""
    
    @staticmethod
    def detect_cycles(spheres: np.ndarray, 
                     cylinders: List[Tuple], 
                     links: List[Tuple],
                     cycle_length: int = 4) -> List[List[int]]:
        """
        Detect cyclic structures in the connectivity graph.
        
        Args:
            spheres: Sphere coordinates
            cylinders: Cylinder coordinates
            links: Connectivity data
            cycle_length: Target cycle length (default: 4 for tetramers)
            
        Returns:
            List of detected cycles (as sphere index lists)
        """
        # Build connectivity graph
        sphere_connections = defaultdict(set)
        
        for sphere_idx, cylinder_idx in links:
            if sphere_idx < len(spheres) and cylinder_idx < len(cylinders):
                for other_sphere_idx, other_cylinder_idx in links:
                    if other_cylinder_idx == cylinder_idx and other_sphere_idx != sphere_idx:
                        sphere_connections[sphere_idx].add(other_sphere_idx)
        
        # Search for 4-vertex cycles
        cycles = []
        visited_sets = set()
        
        for start_sphere in range(len(spheres)):
            if len(sphere_connections[start_sphere]) >= 2:
                neighbors = list(sphere_connections[start_sphere])
                for i, n1 in enumerate(neighbors):
                    for n2 in neighbors[i+1:]:
                        common = sphere_connections[n1] & sphere_connections[n2]
                        if common:
                            for n3 in common:
                                if n3 != start_sphere:
                                    cycle = tuple(sorted([start_sphere, n1, n2, n3]))
                                    if cycle not in visited_sets:
                                        cycles.append(list(cycle))
                                        visited_sets.add(cycle)
        
        return cycles


class Visualizer:
    """Generates 3D visualizations of molecular structures."""
    
    def __init__(self, data: SimulationData):
        """
        Initialize visualizer with simulation data.
        
        Args:
            data: SimulationData object containing structure information
        """
        self.data = data
        self.depth_shader = DepthShading()
        self.structure_detector = StructureDetector()
    
    def _setup_axes(self, ax, elev: float, azim: float, add_margin: float = 0.1):
        """
        Configure 3D axes with appropriate limits and styling.
        
        Args:
            ax: Matplotlib 3D axes object
            elev: Elevation angle
            azim: Azimuth angle
            add_margin: Fractional margin to add around structure (0.1 = 10%)
        """
        spheres = self.data.spheres
        
        # Add margins around structure
        x_margin = (spheres[:, 0].max() - spheres[:, 0].min()) * add_margin
        y_margin = (spheres[:, 1].max() - spheres[:, 1].min()) * add_margin
        z_margin = (spheres[:, 2].max() - spheres[:, 2].min()) * add_margin
        
        ax.set_xlim(spheres[:, 0].min() - x_margin, spheres[:, 0].max() + x_margin)
        ax.set_ylim(spheres[:, 1].min() - y_margin, spheres[:, 1].max() + y_margin)
        ax.set_zlim(spheres[:, 2].min() - z_margin, spheres[:, 2].max() + z_margin)
        
        # Set viewpoint
        ax.view_init(elev=elev, azim=azim)
        
        # Configure axis labels
        ax.set_xlabel('X (Å)', fontsize=12, fontweight='bold', labelpad=10)
        ax.set_ylabel('Y (Å)', fontsize=12, fontweight='bold', labelpad=10)
        ax.set_zlabel('Z (Å)', fontsize=12, fontweight='bold', labelpad=10)
        
        # Configure grid
        ax.grid(True, alpha=0.15, linestyle='--', linewidth=0.5)
        
        # Configure background panes
        ax.xaxis.pane.fill = True
        ax.yaxis.pane.fill = True
        ax.zaxis.pane.fill = True
        ax.xaxis.pane.set_facecolor('white')
        ax.yaxis.pane.set_facecolor('white')
        ax.zaxis.pane.set_facecolor('white')
        ax.xaxis.pane.set_alpha(0.1)
        ax.yaxis.pane.set_alpha(0.1)
        ax.zaxis.pane.set_alpha(0.1)
    
    def create_single_view(self, 
                          output_filename: str,
                          elev: float = 25,
                          azim: float = 45,
                          show_connections: bool = False,
                          use_depth_shading: bool = True,
                          sphere_size: float = 60,
                          cylinder_width: float = 1.8,
                          figsize: Tuple[int, int] = (16, 13),
                          dpi: int = 300):
        """
        Generate a single-viewpoint visualization.
        
        Args:
            output_filename: Output file path
            elev: Elevation angle in degrees
            azim: Azimuth angle in degrees
            show_connections: Whether to display connection lines
            use_depth_shading: Whether to apply depth-based coloring
            sphere_size: Sphere marker size
            cylinder_width: Cylinder line width
            figsize: Figure size in inches
            dpi: Output resolution
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        spheres = self.data.spheres
        cylinders = self.data.cylinders
        links = self.data.links
        
        # Calculate depth-based properties if enabled
        if use_depth_shading:
            sphere_depths = self.depth_shader.calculate_depths(spheres, elev, azim)
            sphere_alphas = self.depth_shader.get_alphas(sphere_depths, 0.35, 0.9)
            sphere_colors = self.depth_shader.get_colors(sphere_depths, 'Reds', 0.5, 1.0)
        else:
            sphere_alphas = [0.7] * len(spheres)
            sphere_colors = 'red'
        
        # Render spheres
        for i, (pos, alpha) in enumerate(zip(spheres, sphere_alphas)):
            color = sphere_colors[i] if use_depth_shading else sphere_colors
            ax.scatter(pos[0], pos[1], pos[2],
                      c=[color] if use_depth_shading else color,
                      marker='o',
                      s=sphere_size,
                      alpha=alpha,
                      edgecolors='darkred',
                      linewidths=0.5,
                      depthshade=True)
        
        # Calculate cylinder depth properties
        cylinder_points = []
        for p1, p2 in cylinders:
            cylinder_points.append(p1)
            cylinder_points.append(p2)
        cylinder_points = np.array(cylinder_points)
        
        if use_depth_shading and len(cylinder_points) > 0:
            cyl_depths = self.depth_shader.calculate_depths(cylinder_points, elev, azim)
            cyl_alphas = self.depth_shader.get_alphas(cyl_depths, 0.25, 0.75)
            cyl_colors = self.depth_shader.get_colors(cyl_depths, 'Blues', 0.5, 1.0)
        else:
            cyl_alphas = [0.5] * len(cylinders)
            cyl_colors = None
        
        # Render cylinders
        for i, (p1, p2) in enumerate(cylinders):
            alpha_val = cyl_alphas[i*2] if use_depth_shading else 0.5
            color_val = cyl_colors[i*2] if use_depth_shading else '#4169E1'
            
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                   color=color_val if not isinstance(color_val, float) else '#4169E1',
                   linewidth=cylinder_width,
                   alpha=alpha_val,
                   solid_capstyle='round')
            
            ax.scatter([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                      c=[cyl_colors[i*2]] if use_depth_shading else '#0047AB',
                      marker='s',
                      s=sphere_size * 0.4,
                      alpha=alpha_val * 0.8,
                      edgecolors='navy',
                      linewidths=0.3,
                      depthshade=True)
        
        # Render connection lines if requested
        if show_connections:
            link_drawn = set()
            for sphere_idx, cylinder_idx in links:
                if sphere_idx < len(spheres) and cylinder_idx < len(cylinders):
                    link_key = (sphere_idx, cylinder_idx)
                    if link_key not in link_drawn:
                        sphere_pos = spheres[sphere_idx]
                        cyl_p1, cyl_p2 = cylinders[cylinder_idx]
                        dist1 = np.linalg.norm(np.array(sphere_pos) - np.array(cyl_p1))
                        dist2 = np.linalg.norm(np.array(sphere_pos) - np.array(cyl_p2))
                        cyl_pos = cyl_p1 if dist1 < dist2 else cyl_p2
                        
                        ax.plot([sphere_pos[0], cyl_pos[0]],
                               [sphere_pos[1], cyl_pos[1]],
                               [sphere_pos[2], cyl_pos[2]],
                               'gray', linewidth=0.3, alpha=0.1)
                        link_drawn.add(link_key)
        
        # Configure axes
        self._setup_axes(ax, elev, azim)
        
        # Add legend
        legend_elements = [
            Patch(facecolor='red', alpha=0.7, label=f'Si-molecules (n={len(spheres)})'),
            Patch(facecolor='blue', alpha=0.5, label=f'Linkers (n={len(cylinders)})')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=11,
                 framealpha=0.9, edgecolor='gray')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_filename, dpi=dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"✓ Saved: {output_filename}")
        plt.close()
    
    def create_multi_view(self, 
                         output_filename: str,
                         dpi: int = 400):
        """
        Generate a 4-panel multi-viewpoint figure.
        
        Args:
            output_filename: Output file path
            dpi: Output resolution
        """
        fig = plt.figure(figsize=(20, 16))
        
        # Define four standard viewpoints
        views = [
            {'elev': 20, 'azim': 45, 'title': '(a) Perspective View', 'pos': 221},
            {'elev': 0, 'azim': 0, 'title': '(b) Front View (YZ plane)', 'pos': 222},
            {'elev': 90, 'azim': 0, 'title': '(c) Top View (XY plane)', 'pos': 223},
            {'elev': 0, 'azim': 90, 'title': '(d) Side View (XZ plane)', 'pos': 224},
        ]
        
        spheres = self.data.spheres
        cylinders = self.data.cylinders
        
        for view in views:
            ax = fig.add_subplot(view['pos'], projection='3d')
            
            # Calculate depth properties for this viewpoint
            sphere_depths = self.depth_shader.calculate_depths(spheres, view['elev'], view['azim'])
            sphere_alphas = self.depth_shader.get_alphas(sphere_depths, 0.35, 0.85)
            sphere_colors = self.depth_shader.get_colors(sphere_depths, 'Reds', 0.5, 1.0)
            
            # Render spheres
            for i, (pos, alpha, color) in enumerate(zip(spheres, sphere_alphas, sphere_colors)):
                ax.scatter(pos[0], pos[1], pos[2],
                          c=[color],
                          marker='o',
                          s=70,
                          alpha=alpha,
                          edgecolors='darkred',
                          linewidths=0.6,
                          depthshade=True)
            
            # Calculate cylinder depth properties
            cylinder_points = []
            for p1, p2 in cylinders:
                cylinder_points.append(p1)
            cylinder_points = np.array(cylinder_points)
            
            if len(cylinder_points) > 0:
                cyl_depths = self.depth_shader.calculate_depths(cylinder_points, view['elev'], view['azim'])
                cyl_alphas = self.depth_shader.get_alphas(cyl_depths, 0.3, 0.7)
                cyl_colors = self.depth_shader.get_colors(cyl_depths, 'Blues', 0.5, 1.0)
            
            # Render cylinders
            for i, (p1, p2) in enumerate(cylinders):
                alpha_val = cyl_alphas[i] if len(cylinder_points) > 0 else 0.5
                color_val = cyl_colors[i] if len(cylinder_points) > 0 else '#4169E1'
                
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                       color=color_val,
                       linewidth=2.2,
                       alpha=alpha_val,
                       solid_capstyle='round')
                
                ax.scatter([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                          c=[color_val],
                          marker='s',
                          s=30,
                          alpha=alpha_val * 0.8,
                          edgecolors='navy',
                          linewidths=0.4,
                          depthshade=True)
            
            # Configure axes
            self._setup_axes(ax, view['elev'], view['azim'], add_margin=0.08)
            ax.set_title(view['title'], fontsize=14, fontweight='bold', pad=15)
            
            # Add legend to first panel only
            if view['pos'] == 221:
                legend_elements = [
                    Patch(facecolor='red', alpha=0.7, label='Si-molecules'),
                    Patch(facecolor='blue', alpha=0.5, label='Linkers')
                ]
                ax.legend(handles=legend_elements, loc='upper left', fontsize=11,
                         framealpha=0.95, edgecolor='lightgray')
        
        plt.tight_layout()
        plt.savefig(output_filename, dpi=dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"✓ Multi-view figure saved: {output_filename}")
        plt.close()
    
    def create_zoom_view(self,
                        center_idx: int,
                        output_filename: str,
                        radius: float = 25.0,
                        elev: float = 25,
                        azim: float = 45,
                        dpi: int = 350):
        """
        Generate a zoomed view focused on a specific region.
        
        Args:
            center_idx: Index of sphere at focus center
            output_filename: Output file path
            radius: Zoom radius in Angstroms
            elev: Elevation angle
            azim: Azimuth angle
            dpi: Output resolution
        """
        spheres = self.data.spheres
        cylinders = self.data.cylinders
        links = self.data.links
        
        center = spheres[center_idx]
        
        # Extract objects within zoom radius
        distances = np.linalg.norm(spheres - center, axis=1)
        sphere_mask = distances <= radius
        sphere_indices = np.where(sphere_mask)[0]
        
        cylinder_indices = []
        for i, (p1, p2) in enumerate(cylinders):
            center_cyl = np.array([(p1[0]+p2[0])/2, (p1[1]+p2[1])/2, (p1[2]+p2[2])/2])
            if np.linalg.norm(center_cyl - center) <= radius * 1.2:
                cylinder_indices.append(i)
        
        # Create figure
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Configure background
        ax.xaxis.pane.fill = True
        ax.yaxis.pane.fill = True
        ax.zaxis.pane.fill = True
        ax.xaxis.pane.set_facecolor('#FAFAFA')
        ax.yaxis.pane.set_facecolor('#FAFAFA')
        ax.zaxis.pane.set_facecolor('#FAFAFA')
        ax.xaxis.pane.set_alpha(0.3)
        ax.yaxis.pane.set_alpha(0.3)
        ax.zaxis.pane.set_alpha(0.3)
        
        # Mark center point
        ax.scatter(center[0], center[1], center[2],
                  c='yellow', marker='*', s=500, alpha=1.0,
                  edgecolors='orange', linewidths=3,
                  label='Focus center', zorder=100)
        
        # Render spheres in zoom region
        for idx in sphere_indices:
            if idx != center_idx:
                pos = spheres[idx]
                dist = np.linalg.norm(pos - center)
                alpha = 0.9 - (dist / radius) * 0.5
                
                ax.scatter(pos[0], pos[1], pos[2],
                          c='red', marker='o', s=150,
                          alpha=alpha,
                          edgecolors='darkred', linewidths=1.0,
                          depthshade=True)
        
        # Render cylinders in zoom region
        for idx in cylinder_indices:
            p1, p2 = cylinders[idx]
            center_cyl = np.array([(p1[0]+p2[0])/2, (p1[1]+p2[1])/2, (p1[2]+p2[2])/2])
            dist = np.linalg.norm(center_cyl - center)
            alpha = 0.8 - (dist / radius) * 0.4
            
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                   color='#4169E1', linewidth=3.5, alpha=alpha,
                   solid_capstyle='round')
            
            ax.scatter([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                      c='#0047AB', marker='s', s=80,
                      alpha=alpha * 0.9,
                      edgecolors='navy', linewidths=0.8,
                      depthshade=True)
        
        # Render connection lines
        for sphere_idx, cylinder_idx in links:
            if sphere_idx in sphere_indices and cylinder_idx in cylinder_indices:
                sphere_pos = spheres[sphere_idx]
                cyl_p1, cyl_p2 = cylinders[cylinder_idx]
                dist1 = np.linalg.norm(sphere_pos - np.array(cyl_p1))
                dist2 = np.linalg.norm(sphere_pos - np.array(cyl_p2))
                cyl_pos = cyl_p1 if dist1 < dist2 else cyl_p2
                
                ax.plot([sphere_pos[0], cyl_pos[0]],
                       [sphere_pos[1], cyl_pos[1]],
                       [sphere_pos[2], cyl_pos[2]],
                       'green', linewidth=1.5, alpha=0.4, linestyle='--')
        
        # Configure axes
        ax.set_xlabel('X (Å)', fontsize=13, fontweight='bold', labelpad=10)
        ax.set_ylabel('Y (Å)', fontsize=13, fontweight='bold', labelpad=10)
        ax.set_zlabel('Z (Å)', fontsize=13, fontweight='bold', labelpad=10)
        
        ax.view_init(elev=elev, azim=azim)
        
        ax.set_xlim(center[0] - radius*1.1, center[0] + radius*1.1)
        ax.set_ylim(center[1] - radius*1.1, center[1] + radius*1.1)
        ax.set_zlim(center[2] - radius*1.1, center[2] + radius*1.1)
        
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
        
        # Add legend
        legend_elements = [
            Patch(facecolor='yellow', alpha=1.0, edgecolor='orange', label='Focus center'),
            Patch(facecolor='red', alpha=0.8, label='Si-molecules'),
            Patch(facecolor='blue', alpha=0.7, label='Linkers'),
            Patch(facecolor='green', alpha=0.4, label='Si-O bonds')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=11,
                 framealpha=0.95, edgecolor='gray')
        
        plt.tight_layout()
        plt.savefig(output_filename, dpi=dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"✓ Saved zoom view: {output_filename}")
        plt.close()
    
    def create_animation(self,
                        output_filename: str,
                        num_frames: int = 36,
                        fps: int = 10,
                        elev: float = 25):
        """
        Generate a 360-degree rotation animation.
        
        Args:
            output_filename: Output file path (.gif)
            num_frames: Number of frames in animation
            fps: Frames per second
            elev: Fixed elevation angle
        """
        try:
            from PIL import Image
        except ImportError:
            print("⚠ Warning: PIL not available. Skipping animation.")
            return
        
        print(f"Creating animation with {num_frames} frames...")
        
        frames = []
        temp_dir = "/tmp/anim_frames"
        os.makedirs(temp_dir, exist_ok=True)
        
        # First pass: create all frames
        temp_files = []
        for i in range(num_frames):
            azim = i * (360.0 / num_frames)
            temp_file = f"{temp_dir}/frame_{i:03d}.png"
            
            self.create_single_view(
                temp_file,
                elev=elev,
                azim=azim,
                use_depth_shading=True,
                sphere_size=60,
                cylinder_width=1.8,
                figsize=(12, 10),
                dpi=150
            )
            
            temp_files.append(temp_file)
            print(f"  Frame {i+1}/{num_frames} completed", end='\r')
        
        print()
        
        # Second pass: load all images and find maximum dimensions
        print("Processing frames for uniform size...")
        raw_frames = [Image.open(f) for f in temp_files]
        
        # Find the maximum dimensions
        max_width = max(img.width for img in raw_frames)
        max_height = max(img.height for img in raw_frames)
        
        # Resize all frames to the same size
        for img in raw_frames:
            # Create a new image with max dimensions and white background
            new_img = Image.new('RGB', (max_width, max_height), 'white')
            # Paste the original image centered
            offset_x = (max_width - img.width) // 2
            offset_y = (max_height - img.height) // 2
            new_img.paste(img, (offset_x, offset_y))
            frames.append(new_img)
        
        # Save as GIF
        frames[0].save(
            output_filename,
            save_all=True,
            append_images=frames[1:],
            duration=1000//fps,
            loop=0
        )
        
        # Clean up temporary files
        import shutil
        shutil.rmtree(temp_dir)
        
        print(f"✓ Animation saved: {output_filename}")
    
    def find_dense_regions(self, num_regions: int = 3) -> List[int]:
        """
        Identify regions with high molecular density.
        
        Args:
            num_regions: Number of dense regions to find
            
        Returns:
            List of sphere indices at region centers
        """
        try:
            from scipy.spatial import distance_matrix
            
            spheres = self.data.spheres
            dist_matrix = distance_matrix(spheres, spheres)
            
            # Count neighbors within 30 Angstroms
            threshold = 30.0
            neighbor_counts = np.sum(dist_matrix < threshold, axis=1) - 1
            
            dense_indices = np.argsort(neighbor_counts)[::-1]
            
            # Select regions that are well-separated from each other
            centers = []
            for idx in dense_indices:
                if len(centers) == 0:
                    centers.append(idx)
                else:
                    min_dist = min([np.linalg.norm(spheres[idx] - spheres[c]) for c in centers])
                    if min_dist > 40.0:
                        centers.append(idx)
                        if len(centers) >= num_regions:
                            break
            
            return centers
        except ImportError:
            # Fallback to random selection if scipy unavailable
            return list(np.random.choice(len(self.data.spheres), 
                                        size=min(num_regions, len(self.data.spheres)),
                                        replace=False))


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='3D Visualization Tool for Macrocycle Formation Structures',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all view types
  python macrocycle_visualizer.py --input data.txt --all
  
  # Generate standard views only
  python macrocycle_visualizer.py --input data.txt --standard-views
  
  # Generate multi-view figure only
  python macrocycle_visualizer.py --input data.txt --multi-view
  
  # Generate zoom views only
  python macrocycle_visualizer.py --input data.txt --zoom-only
  
  # Custom parameters
  python macrocycle_visualizer.py --input data.txt --standard-views --sphere-size 50 --dpi 400
        """
    )
    
    # Basic arguments
    parser.add_argument('--input', default='/mnt/user-data/uploads/output.txt',
                       help='Input file path')
    parser.add_argument('--output-dir', default='/mnt/user-data/outputs/visualizations',
                       help='Output directory')
    
    # Generation options
    parser.add_argument('--all', action='store_true',
                       help='Generate all view types')
    parser.add_argument('--standard-views', action='store_true',
                       help='Generate standard single views')
    parser.add_argument('--multi-view', action='store_true',
                       help='Generate 4-panel multi-view figure')
    parser.add_argument('--zoom-only', action='store_true',
                       help='Generate zoom views only')
    parser.add_argument('--animation', action='store_true',
                       help='Generate rotation animation')
    
    # Customization options
    parser.add_argument('--sphere-size', type=float, default=60,
                       help='Sphere marker size (default: 60)')
    parser.add_argument('--cylinder-width', type=float, default=1.8,
                       help='Cylinder line width (default: 1.8)')
    parser.add_argument('--dpi', type=int, default=300,
                       help='Image resolution (default: 300)')
    parser.add_argument('--zoom-radius', type=float, default=25.0,
                       help='Zoom radius in Angstroms (default: 25.0)')
    parser.add_argument('--num-zoom-regions', type=int, default=3,
                       help='Number of zoom regions (default: 3)')
    
    args = parser.parse_args()
    
    # Default behavior if no options specified
    if not any([args.all, args.standard_views, args.multi_view, 
                args.zoom_only, args.animation]):
        args.all = True
    
    # Print header
    print("\n" + "="*70)
    print("MACROCYCLE FORMATION 3D VISUALIZATION TOOL")
    print("="*70)
    
    # Load data
    print(f"\nLoading structure data from: {args.input}")
    try:
        data = SimulationData(args.input)
    except FileNotFoundError:
        print(f"✗ Error: File not found: {args.input}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        sys.exit(1)
    
    print(f"✓ Loaded {data.num_spheres} spheres and {data.num_cylinders} cylinders")
    
    # Display statistics
    data.print_statistics()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create visualizer
    viz = Visualizer(data)
    
    # Generate standard views
    if args.all or args.standard_views:
        print("\n" + "="*70)
        print("GENERATING STANDARD VIEWS")
        print("="*70)
        
        views = [
            {'name': 'default', 'elev': 25, 'azim': 45, 'desc': 'Default perspective'},
            {'name': 'front', 'elev': 0, 'azim': 0, 'desc': 'Front view'},
            {'name': 'side', 'elev': 0, 'azim': 90, 'desc': 'Side view'},
            {'name': 'top', 'elev': 90, 'azim': 0, 'desc': 'Top view'},
            {'name': 'diagonal', 'elev': 35, 'azim': 135, 'desc': 'Diagonal view'},
        ]
        
        for i, view in enumerate(views, 1):
            print(f"\n[{i}/{len(views)}] {view['desc']}...")
            viz.create_single_view(
                f"{args.output_dir}/view_{view['name']}.png",
                elev=view['elev'],
                azim=view['azim'],
                show_connections=False,
                use_depth_shading=True,
                sphere_size=args.sphere_size,
                cylinder_width=args.cylinder_width,
                dpi=args.dpi
            )
        
        # Version with connections
        print(f"\n[{len(views)+1}/{len(views)+1}] With connections...")
        viz.create_single_view(
            f"{args.output_dir}/view_with_connections.png",
            elev=25,
            azim=45,
            show_connections=True,
            use_depth_shading=True,
            sphere_size=args.sphere_size,
            cylinder_width=args.cylinder_width,
            dpi=args.dpi
        )
    
    # Generate multi-view figure
    if args.all or args.multi_view:
        print("\n" + "="*70)
        print("GENERATING MULTI-VIEW FIGURE")
        print("="*70)
        viz.create_multi_view(
            f"{args.output_dir}/multi_view_4panel.png",
            dpi=400
        )
    
    # Generate zoom views
    if args.all or args.zoom_only:
        print("\n" + "="*70)
        print("GENERATING ZOOM VIEWS")
        print("="*70)
        
        print("\nIdentifying dense regions...")
        dense_centers = viz.find_dense_regions(num_regions=args.num_zoom_regions)
        print(f"Found {len(dense_centers)} regions")
        
        for i, center_idx in enumerate(dense_centers, 1):
            print(f"\n[{i}/{len(dense_centers)}] Creating zoom views for region {center_idx}...")
            
            views = [
                {'elev': 25, 'azim': 45, 'suffix': 'perspective'},
                {'elev': 90, 'azim': 0, 'suffix': 'top'},
            ]
            
            for view in views:
                viz.create_zoom_view(
                    center_idx,
                    f"{args.output_dir}/zoom_region_{i}_{view['suffix']}.png",
                    radius=args.zoom_radius,
                    elev=view['elev'],
                    azim=view['azim'],
                    dpi=350
                )
    
    # Generate animation
    if args.all or args.animation:
        print("\n" + "="*70)
        print("GENERATING ANIMATION")
        print("="*70)
        viz.create_animation(
            f"{args.output_dir}/rotation_animation.gif",
            num_frames=36,
            fps=10,
            elev=25
        )
    
    # Completion message
    print("\n" + "="*70)
    print("✓ VISUALIZATION COMPLETED")
    print("="*70)
    print(f"\nAll files saved to: {args.output_dir}")
    print("\nGenerated files:")
    
    # List output files
    import glob
    files = sorted(glob.glob(f"{args.output_dir}/*"))
    for f in files:
        size = os.path.getsize(f) / (1024 * 1024)
        print(f"  - {os.path.basename(f)} ({size:.1f} MB)")
    
    print("\nKey features:")
    print("  • Optimized object sizes for better visibility")
    print("  • Depth-based transparency and color shading")
    print("  • Increased margins around structures")
    print("  • Reduced line widths for cleaner appearance")
    print("  • Multiple viewpoints for comprehensive analysis")
    if args.zoom_only or args.all:
        print("  • Detailed zoom views of dense regions")
    if args.all or args.animation:
        print("  • 360° rotation animation")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()