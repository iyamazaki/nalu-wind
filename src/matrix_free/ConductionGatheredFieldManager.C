// Copyright 2017 National Technology & Engineering Solutions of Sandia, LLC
// (NTESS), National Renewable Energy Laboratory, University of Texas Austin,
// Northwest Research Associates. Under the terms of Contract DE-NA0003525
// with NTESS, the U.S. Government retains certain rights in this software.
//
// This software is released under the BSD 3-clause license. See LICENSE file
// for more details.
//

#include "matrix_free/ConductionGatheredFieldManager.h"

#include <stk_mesh/base/Types.hpp>

#include "matrix_free/ConductionFields.h"
#include "matrix_free/PolynomialOrders.h"
#include "matrix_free/LinearExposedAreas.h"
#include "matrix_free/StkSimdConnectivityMap.h"
#include "matrix_free/StkSimdFaceConnectivityMap.h"
#include "matrix_free/StkSimdGatheredElementData.h"
#include "matrix_free/StkSimdNodeConnectivityMap.h"
#include "matrix_free/KokkosFramework.h"
#include "stk_mesh/base/MetaData.hpp"

#include "stk_mesh/base/NgpProfilingBlock.hpp"

namespace sierra {
namespace nalu {
namespace matrix_free {

template <int p>
ConductionGatheredFieldManager<p>::ConductionGatheredFieldManager(
  stk::mesh::BulkData& bulk_in,
  stk::mesh::Selector active_in,
  stk::mesh::Selector dirichlet_in,
  stk::mesh::Selector flux_in)
  : bulk(bulk_in),
    meta(bulk_in.mesh_meta_data()),
    active(active_in),
    conn(stk_connectivity_map<p>(bulk.get_updated_ngp_mesh(), active)),
    dirichlet(dirichlet_in),
    dirichlet_nodes(simd_node_map(bulk.get_updated_ngp_mesh(), dirichlet)),
    flux(flux_in),
    flux_faces(face_node_map<p>(bulk.get_updated_ngp_mesh(), flux_in))
{
}

template <int p>
void
ConductionGatheredFieldManager<p>::gather_all()
{
  stk::mesh::ProfilingBlock pf("ConductionGatheredFieldManager<p>::gather_all");
  fields = gather_required_conduction_fields<p>(meta, conn);
  coefficient_fields.volume_metric = fields.volume_metric;
  coefficient_fields.diffusion_metric = fields.diffusion_metric;

  if (dirichlet_nodes.extent_int(0) > 0) {
    bc_fields.qp1 =
      node_scalar_view("qp1_at_bc", dirichlet_nodes.extent_int(0));
    stk_simd_scalar_node_gather(
      dirichlet_nodes, get_ngp_field<double>(meta, conduction_info::q_name),
      bc_fields.qp1);

    bc_fields.qbc =
      node_scalar_view("qspecified_at_bc", dirichlet_nodes.extent_int(0));
    stk_simd_scalar_node_gather(
      dirichlet_nodes, get_ngp_field<double>(meta, conduction_info::qbc_name),
      bc_fields.qbc);
  }

  if (flux_faces.extent_int(0) > 0) {
    {
      auto face_coords =
        face_vector_view<p>("face_coords", flux_faces.extent_int(0));
      stk_simd_face_vector_field_gather<p>(
        flux_faces, get_ngp_field<double>(meta, conduction_info::coord_name),
        face_coords);
      flux_fields.exposed_areas = geom::exposed_areas<p>(face_coords);
    }
    flux_fields.flux = face_scalar_view<p>("flux", flux_faces.extent_int(0));
    stk_simd_face_scalar_field_gather<p>(
      flux_faces, get_ngp_field<double>(meta, conduction_info::flux_name),
      flux_fields.flux);
  }
}

template <int p>
void
ConductionGatheredFieldManager<p>::update_solution_fields()
{
  stk::mesh::ProfilingBlock pf(
    "ConductionGatheredFieldManager<p>::update_solution_fields");
  stk_simd_scalar_field_gather<p>(
    conn, get_ngp_field<double>(meta, conduction_info::q_name), fields.qp1);

  if (dirichlet_nodes.extent_int(0) > 0) {
    stk_simd_scalar_node_gather(
      dirichlet_nodes, get_ngp_field<double>(meta, conduction_info::q_name),
      bc_fields.qp1);
  }
}

template <int p>
void
ConductionGatheredFieldManager<p>::swap_states()
{
  stk::mesh::ProfilingBlock pf(
    "ConductionGatheredFieldManager<p>::swap_states");
  auto qm1 = fields.qm1;
  fields.qm1 = fields.qp0;
  fields.qp0 = fields.qp1;
  fields.qp1 = qm1;
}
INSTANTIATE_POLYCLASS(ConductionGatheredFieldManager);

} // namespace matrix_free
} // namespace nalu
} // namespace sierra
