# nhlib: A New Hazard Library
# Copyright (C) 2012 GEM Foundation
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Module :mod:`nhlib.geo.surface.complex_fault` defines
:class:`ComplexFaultSurface`.
"""
from nhlib.geo.line import Line
from nhlib.geo.surface.base import BaseSurface
from nhlib.geo.mesh import RectangularMesh
from nhlib.geo._utils import ensure


class ComplexFaultSurface(BaseSurface):
    """
    Represent a complex fault surface as 3D mesh of points (not necessarily
    uniformly spaced across the surface area).

    :param mesh:
        Instance of :class:`~nhlib.geo.mesh.RectangularMesh` representing
        surface geometry.

    Another way to construct the surface object is to call
    :meth:`from_fault_data`.
    """
    def __init__(self, mesh):
        super(ComplexFaultSurface, self).__init__()
        self.mesh = mesh
        assert not 1 in self.mesh.shape
        self.strike = self.dip = None

    def _create_mesh(self):
        """
        Return a mesh provided to object's constructor.
        """
        return self.mesh

    def get_dip(self):
        """
        Return the fault dip as the average dip over the mesh.

        The average dip is defined as the weighted mean inclination
        of all the mesh cells. See
        :meth:`nhlib.geo.mesh.RectangularMesh.get_mean_inclination_and_azimuth`

        :returns:
            The average dip, in decimal degrees.
        """
        # uses the same approach as in simple fault surface
        if self.dip is None:
            mesh = self.get_mesh()
            self.dip, self.strike = mesh.get_mean_inclination_and_azimuth()
        return self.dip

    def get_strike(self):
        """
        Return the fault strike as the average strike over the mesh.

        The average strike is defined as the weighted mean azimuth
        of all the mesh cells. See
        :meth:`nhlib.geo.mesh.RectangularMesh.get_mean_inclination_and_azimuth`

        :returns:
            The average strike, in decimal degrees.
        """
        if self.strike is None:
            self.get_dip()  # this should cache strike value
        return self.strike

    @classmethod
    def check_fault_data(cls, edges, mesh_spacing):
        """
        Verify the fault data and raise ``ValueError`` if anything is wrong.

        This method doesn't have to be called by hands before creating the
        surface object, because it is called from :meth:`from_fault_data`.
        """
        ensure(len(edges) >= 2, "at least two edges are required")
        ensure(all(len(edge) >= 2 for edge in edges),
               "at least two points must be defined in each edge")
        ensure(mesh_spacing > 0.0, "mesh spacing must be positive")
        # TODO: more strict/sophisticated checks for edges?

    @classmethod
    def from_fault_data(cls, edges, mesh_spacing):
        """
        Create and return a fault surface using fault source data.

        :param edges:
            A list of at least two horizontal edges of the surface
            as instances of :class:`nhlib.geo.line.Line`. The list
            should be in top-to-bottom order (the shallowest edge
            first).
        :param mesh_spacing:
            Distance between two subsequent points in a mesh, in km.
        :returns:
            An instance of :class:`ComplexFaultSurface` created using
            that data.
        :raises ValueError:
            If requested mesh spacing is too big for the surface geometry
            (doesn't allow to put a single mesh cell along length and/or
            width).

        Uses :meth:`check_fault_data` for checking parameters.
        """
        cls.check_fault_data(edges, mesh_spacing)

        edges_lengths = [edge.get_length() for edge in edges]
        mean_length = sum(edges_lengths) / len(edges)
        num_hor_points = int(round(mean_length / mesh_spacing)) + 1
        if num_hor_points <= 1:
            raise ValueError(
                'mesh spacing %.1f km is to big for mean length %.1f km' %
                (mesh_spacing, mean_length)
            )
        edges = [edge.resample_to_num_points(num_hor_points).points
                 for i, edge in enumerate(edges)]

        vert_edges = [Line(v_edge) for v_edge in zip(*edges)]
        vert_edges_lengths = [v_edge.get_length() for v_edge in vert_edges]
        mean_width = sum(vert_edges_lengths) / (num_hor_points)
        num_vert_points = int(round(mean_width / mesh_spacing)) + 1
        if num_vert_points <= 1:
            raise ValueError(
                'mesh spacing %.1f km is to big for mean width %.1f km' %
                (mesh_spacing, mean_width)
            )

        points = zip(*[v_edge.resample_to_num_points(num_vert_points).points
                       for v_edge in vert_edges])
        mesh = RectangularMesh.from_points_list(points)
        assert 1 not in mesh.shape
        return cls(mesh)
