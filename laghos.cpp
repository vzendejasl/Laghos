// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.
//
//                     __                __
//                    / /   ____  ____  / /_  ____  _____
//                   / /   / __ `/ __ `/ __ \/ __ \/ ___/
//                  / /___/ /_/ / /_/ / / / / /_/ (__  )
//                 /_____/\__,_/\__, /_/ /_/\____/____/
//                             /____/
//
//             High-order Lagrangian Hydrodynamics Miniapp
//
// Laghos(LAGrangian High-Order Solver) is a miniapp that solves the
// time-dependent Euler equation of compressible gas dynamics in a moving
// Lagrangian frame using unstructured high-order finite element spatial
// discretization and explicit high-order time-stepping. Laghos is based on the
// numerical algorithm described in the following article:
//
//    V. Dobrev, Tz. Kolev and R. Rieben, "High-order curvilinear finite element
//    methods for Lagrangian hydrodynamics", SIAM Journal on Scientific
//    Computing, (34) 2012, pp. B606–B641, https://doi.org/10.1137/120864672.
//
// Test problems:
//    p = 0  --> Taylor-Green vortex (smooth problem).
//    p = 1  --> Sedov blast.
//    p = 2  --> 1D Sod shock tube.
//    p = 3  --> Triple point.
//    p = 4  --> Gresho vortex (smooth problem).
//    p = 5  --> 2D Riemann problem, config. 12 of doi.org/10.1002/num.10025
//    p = 6  --> 2D Riemann problem, config.  6 of doi.org/10.1002/num.10025
//    p = 7  --> 2D Rayleigh-Taylor instability problem.
//
// Sample runs: see README.md, section 'Verification of Results'.
//
// Combinations resulting in 3D uniform Cartesian MPI partitionings of the mesh:
// -m data/cube01_hex.mesh   -pt 211 for  2 / 16 / 128 / 1024 ... tasks.
// -m data/cube_922_hex.mesh -pt 921 for    / 18 / 144 / 1152 ... tasks.
// -m data/cube_522_hex.mesh -pt 522 for    / 20 / 160 / 1280 ... tasks.
// -m data/cube_12_hex.mesh  -pt 311 for  3 / 24 / 192 / 1536 ... tasks.
// -m data/cube01_hex.mesh   -pt 221 for  4 / 32 / 256 / 2048 ... tasks.
// -m data/cube_922_hex.mesh -pt 922 for    / 36 / 288 / 2304 ... tasks.
// -m data/cube_522_hex.mesh -pt 511 for  5 / 40 / 320 / 2560 ... tasks.
// -m data/cube_12_hex.mesh  -pt 321 for  6 / 48 / 384 / 3072 ... tasks.
// -m data/cube01_hex.mesh   -pt 111 for  8 / 64 / 512 / 4096 ... tasks.
// -m data/cube_922_hex.mesh -pt 911 for  9 / 72 / 576 / 4608 ... tasks.
// -m data/cube_522_hex.mesh -pt 521 for 10 / 80 / 640 / 5120 ... tasks.
// -m data/cube_12_hex.mesh  -pt 322 for 12 / 96 / 768 / 6144 ... tasks.
// Example runs:
// mpirun -np 10 ./laghos -p 0 -dim 3 -rs 1 -rp 2 -tf 0.08 -pa -visit -iv -diag output.txt -mach 0.28 -u0 1.0 -s 7 -fv -Re 100 -cfl 0.2
// mpirun -np 8 ./laghos -p 0 -dim 3 -rs 1 -rp 2 -tf 0.3 -pa -visit -iv -diag output.txt -mach 0.28 -u0 1.0 -s 7 -fv -Re 200 -cfl 0.5 --interp-cycle 1
// mpirun -np 8 ./laghos -p 8 -dim 3 -rs 1 -rp 2 -ok 2 -ot 1 -s 2 -tf 0.1 -fv -Re 66.6732 -cond -pr 0.71 -ms 1000 -visit -dt 1e-3
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <sys/time.h>
#include <sys/resource.h>
#include "laghos_solver.hpp"

using std::cout;
using std::cerr;
using std::endl;
using namespace mfem;

// Choice for the problem setup.
static int problem, dim;
static double p0_user = -1.0;
static double mach_number = -1.0;
static double mach_u0 = 1.0;
static double p0_background = 1.0;

// Global bounding box for initialization
Vector bb_min, bb_max;

// Forward declarations.
double e0(const Vector &);
double rho0(const Vector &);
double gamma_func(const Vector &);
void v0(const Vector &, Vector &);

static long GetMaxRssMB();
static void display_banner(std::ostream&);
static void Checks(const int ti, const double norm, int &checks);
static double ComputeEnstrophy(const ParGridFunction &v_gf);
static void SaveParallelMesh(const ParMesh &pmesh, const char *mesh_dir,
                             const char *fname_base);

ParGridFunction* InterpolateFieldPeriodic(ParMesh &src_mesh,
                                          ParGridFunction &src_gf,
                                          ParMesh &tar_mesh,
                                          int fieldtype,
                                          int order,
                                          double Lx,
                                          double Ly,
                                          double Lz,
                                          bool use_periodic);

ParGridFunction* InterpolateFieldPeriodic(ParMesh &src_mesh,
                                          ParGridFunction &src_gf,
                                          ParMesh &tar_mesh,
                                          int fieldtype,
                                          int order,
                                          double Lx,
                                          double Ly,
                                          double Lz,
                                          bool use_periodic);

namespace Diagnostics
{

// Coefficient for 2D Vorticity (scalar)
class VorticityCoefficient : public Coefficient
{
protected:
   ParGridFunction *v_gf;
public:
   VorticityCoefficient(ParGridFunction *v_gf_) : v_gf(v_gf_) { }
   virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      Vector curl(1);
      v_gf->GetCurl(T, curl);
      return curl(0);
   }
};

// Coefficient for 3D Vorticity (vector)
class VorticityVectorCoefficient : public VectorCoefficient
{
protected:
   ParGridFunction *v_gf;
public:
   VorticityVectorCoefficient(int dim, ParGridFunction *v_gf_)
      : VectorCoefficient(dim), v_gf(v_gf_) { }
   virtual void Eval(Vector &V, ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      v_gf->GetCurl(T, V);
   }
};

void ComputeCurl(ParGridFunction &u, ParGridFunction &cu)
{
   ParFiniteElementSpace *fes = u.ParFESpace();
   ParFiniteElementSpace *cfes = cu.ParFESpace();
   int dim = fes->GetMesh()->Dimension();
   int vdim = cfes->GetVDim(); // 1 for 2D, 3 for 3D

   // AccumulateAndCountZones.
   Array<int> zones_per_vdof;
   zones_per_vdof.SetSize(cfes->GetVSize());
   zones_per_vdof = 0;

   cu = 0.0;

   // Local interpolation.
   int elndofs;
   Array<int> vdofs;
   Vector curl(dim == 3 ? 3 : 1);

   for (int e = 0; e < fes->GetNE(); ++e)
   {
      cfes->GetElementVDofs(e, vdofs); // Get vdofs for result
      ElementTransformation *tr = fes->GetElementTransformation(e);
      const FiniteElement *el = fes->GetFE(e);
      elndofs = el->GetDof();

      for (int dof = 0; dof < elndofs; ++dof)
      {
         // Project.
         const IntegrationPoint &ip = el->GetNodes().IntPoint(dof);
         tr->SetIntPoint(&ip);

         u.GetCurl(*tr, curl);

         if (dim == 2)
         {
             // 2D: curl is scalar (1 component), cu is scalar
             int ldof = vdofs[dof];
             cu(ldof) += curl(0);
             zones_per_vdof[ldof]++;
         }
         else
         {
             // 3D: curl is vector (3 components), cu is vector
             // Check ordering. ParFiniteElementSpace default is byNODES
             bool by_nodes = (cfes->GetOrdering() == Ordering::byNODES);
             for (int j = 0; j < vdim; ++j)
             {
                 int ldof = by_nodes ? vdofs[j*elndofs + dof] : vdofs[dof*vdim + j];
                 cu(ldof) += curl(j);
                 zones_per_vdof[ldof]++;
             }
         }
      }
   }

   // Communication
   GroupCommunicator &gcomm = cfes->GroupComm();
   gcomm.Reduce<int>(zones_per_vdof, GroupCommunicator::Sum);
   gcomm.Bcast<int>(zones_per_vdof);

   gcomm.Reduce<double>(cu.GetData(), GroupCommunicator::Sum);
   gcomm.Bcast<double>(cu.GetData());

   // Compute means.
   for (int i = 0; i < cu.Size(); i++)
   {
      const int nz = zones_per_vdof[i];
      if (nz)
      {
         cu(i) /= nz;
      }
   }
}

void ComputeVortexStretching(ParGridFunction &u, ParGridFunction &w,
                             ParGridFunction &w_stretch)
{
   ParFiniteElementSpace *fes = u.ParFESpace();
   ParFiniteElementSpace *wfes = w_stretch.ParFESpace();
   int dim = fes->GetMesh()->Dimension();
   int vdim = wfes->GetVDim();

   u.HostRead();
   w.HostRead();
   w_stretch.HostReadWrite();

   w_stretch = 0.0;

   Array<int> zones_per_vdof;
   zones_per_vdof.SetSize(wfes->GetVSize());
   zones_per_vdof = 0;

   int elndofs;
   Array<int> vdofs;
   DenseMatrix grad_u;
   Vector w_val, stretch_val;

   for (int e = 0; e < fes->GetNE(); ++e)
   {
      wfes->GetElementVDofs(e, vdofs);
      ElementTransformation *tr = fes->GetElementTransformation(e);
      const FiniteElement *el = fes->GetFE(e);
      elndofs = el->GetDof();

      for (int dof = 0; dof < elndofs; ++dof)
      {
         const IntegrationPoint &ip = el->GetNodes().IntPoint(dof);
         tr->SetIntPoint(&ip);

         if (dim == 3)
         {
             u.GetVectorGradient(*tr, grad_u);
             w.GetVectorValue(*tr, ip, w_val);

             stretch_val.SetSize(3);
             grad_u.Mult(w_val, stretch_val);

             bool by_nodes = (wfes->GetOrdering() == Ordering::byNODES);
             for (int j = 0; j < vdim; ++j)
             {
                 int ldof = by_nodes ? vdofs[j*elndofs + dof] : vdofs[dof*vdim + j];
                 w_stretch(ldof) += stretch_val(j);
                 zones_per_vdof[ldof]++;
             }
         }
         else
         {
             // 2D: Stretching is zero
             int ldof = vdofs[dof];
             w_stretch(ldof) += 0.0;
             zones_per_vdof[ldof]++;
         }
      }
   }

   GroupCommunicator &gcomm = wfes->GroupComm();
   gcomm.Reduce<int>(zones_per_vdof, GroupCommunicator::Sum);
   gcomm.Bcast<int>(zones_per_vdof);

   gcomm.Reduce<double>(w_stretch.GetData(), GroupCommunicator::Sum);
   gcomm.Bcast<double>(w_stretch.GetData());

   for (int i = 0; i < zones_per_vdof.Size(); i++)
   {
      const int nz = zones_per_vdof[i];
      if (nz)
      {
         w_stretch(i) /= nz;
      }
   }
}

void ComputeVortexCompression(ParGridFunction &u, ParGridFunction &w,
                              ParGridFunction &w_compress)
{
   ParFiniteElementSpace *fes = u.ParFESpace();
   ParFiniteElementSpace *wfes = w_compress.ParFESpace();
   int dim = fes->GetMesh()->Dimension();
   int vdim = wfes->GetVDim();

   u.HostRead();
   w.HostRead();
   w_compress.HostReadWrite();

   w_compress = 0.0;

   Array<int> zones_per_vdof;
   zones_per_vdof.SetSize(wfes->GetVSize());
   zones_per_vdof = 0;

   int elndofs;
   Array<int> vdofs;
   Vector w_val, compress_val;
   double div_u;

   for (int e = 0; e < fes->GetNE(); ++e)
   {
      wfes->GetElementVDofs(e, vdofs);
      ElementTransformation *tr = fes->GetElementTransformation(e);
      const FiniteElement *el = fes->GetFE(e);
      elndofs = el->GetDof();

      for (int dof = 0; dof < elndofs; ++dof)
      {
         const IntegrationPoint &ip = el->GetNodes().IntPoint(dof);
         tr->SetIntPoint(&ip);

         div_u = u.GetDivergence(*tr);

         if (dim == 2)
         {
             double w_scalar = w.GetValue(*tr);
             double val_compress = -1.0 * w_scalar * div_u;

             int ldof = vdofs[dof];
             w_compress(ldof) += val_compress;
             zones_per_vdof[ldof]++;
         }
         else
         {
             w.GetVectorValue(*tr, ip, w_val);

             compress_val.SetSize(3);
             compress_val = w_val;
             compress_val *= -div_u;

             bool by_nodes = (wfes->GetOrdering() == Ordering::byNODES);
             for (int j = 0; j < vdim; ++j)
             {
                 int ldof = by_nodes ? vdofs[j*elndofs + dof] : vdofs[dof*vdim + j];
                 w_compress(ldof) += compress_val(j);
                 zones_per_vdof[ldof]++;
             }
         }
      }
   }

   GroupCommunicator &gcomm = wfes->GroupComm();
   gcomm.Reduce<int>(zones_per_vdof, GroupCommunicator::Sum);
   gcomm.Bcast<int>(zones_per_vdof);

   gcomm.Reduce<double>(w_compress.GetData(), GroupCommunicator::Sum);
   gcomm.Bcast<double>(w_compress.GetData());

   for (int i = 0; i < zones_per_vdof.Size(); i++)
   {
      const int nz = zones_per_vdof[i];
      if (nz)
      {
         w_compress(i) /= nz;
      }
   }
}

void ProjectL2toH1(ParGridFunction &l2_func, ParGridFunction &h1_func)
{
   ParFiniteElementSpace *h1_fes = h1_func.ParFESpace();

   l2_func.HostRead();
   h1_func.HostReadWrite();

   h1_func = 0.0;

   Array<int> zones_per_vdof;
   zones_per_vdof.SetSize(h1_fes->GetVSize());
   zones_per_vdof = 0;

   Array<int> vdofs;

   for (int e = 0; e < h1_fes->GetNE(); ++e)
   {
      h1_fes->GetElementVDofs(e, vdofs);
      ElementTransformation *tr = h1_fes->GetElementTransformation(e);
      const FiniteElement *el = h1_fes->GetFE(e);
      int elndofs = el->GetDof();

      for (int dof = 0; dof < elndofs; ++dof)
      {
         const IntegrationPoint &ip = el->GetNodes().IntPoint(dof);
         tr->SetIntPoint(&ip);

         double val = l2_func.GetValue(*tr);

         h1_func(vdofs[dof]) += val;
         zones_per_vdof[vdofs[dof]]++;
      }
   }

   GroupCommunicator &gcomm = h1_fes->GroupComm();
   gcomm.Reduce<int>(zones_per_vdof, GroupCommunicator::Sum);
   gcomm.Bcast<int>(zones_per_vdof);

   gcomm.Reduce<double>(h1_func.GetData(), GroupCommunicator::Sum);
   gcomm.Bcast<double>(h1_func.GetData());

   for (int i = 0; i < zones_per_vdof.Size(); i++)
   {
      if (zones_per_vdof[i])
      {
         h1_func(i) /= zones_per_vdof[i];
      }
   }
}

double ComputeConductionSurfaceFlux(ParGridFunction &e_h1,
                                    const double kappa_e)
{
   ParFiniteElementSpace *h1_fes = e_h1.ParFESpace();
   ParMesh *pmesh = h1_fes->GetParMesh();
   const int dim = pmesh->Dimension();

   if (pmesh->GetNBE() == 0 || kappa_e == 0.0)
   {
      return 0.0;
   }

   e_h1.HostRead();

   double local = 0.0;
   Vector grad(dim), nor(dim);

   for (int be = 0; be < pmesh->GetNBE(); ++be)
   {
      FaceElementTransformations *ftr = pmesh->GetBdrFaceTransformations(be);
      if (!ftr) { continue; }

      const FiniteElement *fe = h1_fes->GetFE(ftr->Elem1No);
      const int intorder = 2 * fe->GetOrder();
      const IntegrationRule &ir =
         IntRules.Get(ftr->GetGeometryType(), intorder);

      for (int j = 0; j < ir.GetNPoints(); ++j)
      {
         const IntegrationPoint &ip = ir.IntPoint(j);
         ftr->SetIntPoint(&ip);

         ElementTransformation &tr = ftr->GetElement1Transformation();
         const IntegrationPoint &eip = ftr->GetElement1IntPoint();
         tr.SetIntPoint(&eip);

         e_h1.GetGradient(tr, grad);

         if (dim > 1)
         {
            CalcOrtho(ftr->Face->Jacobian(), nor);
         }
         else
         {
            nor.SetSize(1);
            nor(0) = 1.0;
         }

         const double q_dot_n = -kappa_e * (grad * nor);
         local += ip.weight * q_dot_n;
      }
   }

   double global = 0.0;
   MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, pmesh->GetComm());
   return global;
}

void ComputeBaroclinicTerm(ParGridFunction &rho_h1, ParGridFunction &e_h1,
                           ParGridFunction &w_baroclinic)
{
   ParFiniteElementSpace *h1_fes = rho_h1.ParFESpace();
   ParFiniteElementSpace *wfes = w_baroclinic.ParFESpace();

   rho_h1.HostRead();
   e_h1.HostRead();
   w_baroclinic.HostReadWrite();

   int dim = h1_fes->GetMesh()->Dimension();

   w_baroclinic = 0.0;

   Array<int> zones_per_vdof;
   zones_per_vdof.SetSize(wfes->GetVSize());
   zones_per_vdof = 0;

   Array<int> vdofs;
   Vector grad_rho, grad_e;
   Vector baro_val;

   double gamma = 5.0/3.0;

   for (int e = 0; e < h1_fes->GetNE(); ++e)
   {
      wfes->GetElementVDofs(e, vdofs);
      ElementTransformation *tr = h1_fes->GetElementTransformation(e);
      const FiniteElement *el = h1_fes->GetFE(e);
      int elndofs = el->GetDof();

      for (int dof = 0; dof < elndofs; ++dof)
      {
         const IntegrationPoint &ip = el->GetNodes().IntPoint(dof);
         tr->SetIntPoint(&ip);

         rho_h1.GetGradient(*tr, grad_rho);
         e_h1.GetGradient(*tr, grad_e);

         double rho_val = rho_h1.GetValue(*tr);
         double scale = (gamma - 1.0) / (rho_val + 1e-12);

         if (dim == 2)
         {
             double cross_z = grad_rho(0) * grad_e(1) - grad_rho(1) * grad_e(0);
             w_baroclinic(vdofs[dof]) += scale * cross_z;
             zones_per_vdof[vdofs[dof]]++;
         }
         else
         {
             baro_val.SetSize(3);
             baro_val(0) = grad_rho(1)*grad_e(2) - grad_rho(2)*grad_e(1);
             baro_val(1) = grad_rho(2)*grad_e(0) - grad_rho(0)*grad_e(2);
             baro_val(2) = grad_rho(0)*grad_e(1) - grad_rho(1)*grad_e(0);

             baro_val *= scale;

             bool by_nodes = (wfes->GetOrdering() == Ordering::byNODES);
             int vdim_local = wfes->GetVDim();

             for (int j = 0; j < vdim_local; ++j)
             {
                 int ldof = by_nodes ? vdofs[j*elndofs + dof] : vdofs[dof*vdim_local + j];
                 w_baroclinic(ldof) += baro_val(j);
                 zones_per_vdof[ldof]++;
             }
         }
      }
   }

   GroupCommunicator &gcomm = wfes->GroupComm();
   gcomm.Reduce<int>(zones_per_vdof, GroupCommunicator::Sum);
   gcomm.Bcast<int>(zones_per_vdof);

   gcomm.Reduce<double>(w_baroclinic.GetData(), GroupCommunicator::Sum);
   gcomm.Bcast<double>(w_baroclinic.GetData());

   for (int i = 0; i < zones_per_vdof.Size(); i++)
   {
      if (zones_per_vdof[i]) w_baroclinic(i) /= zones_per_vdof[i];
   }
}

void ComputeJacobian(ParMesh &pmesh,
                     ParGridFunction &detJ_gf,
                     ParGridFunction &logdetJ_gf,
                     ParGridFunction &Jfrob_gf,
                     ParGridFunction &Jinvfrob_gf,
                     ParGridFunction &cond_gf,
                     std::vector<ParGridFunction*> &Jcol_gf,
                     int qorder,
                     double det_eps = 1e-14)
{
   using namespace mfem;
   const int dim = pmesh.Dimension();

   QuadratureSpace qs(&pmesh, qorder);

   // QuadratureFunctions for scalars
   QuadratureFunction detJ_qf(&qs);
   QuadratureFunction logdetJ_qf(&qs);
   QuadratureFunction Jfrob_qf(&qs);
   QuadratureFunction Jinvfrob_qf(&qs);
   QuadratureFunction cond_qf(&qs);

   detJ_qf = 0.0; logdetJ_qf = 0.0; Jfrob_qf = 0.0; Jinvfrob_qf = 0.0; cond_qf = 0.0;

   // QuadratureFunctions for vector columns
   // Use vdim = dim for vector fields
   std::vector<QuadratureFunction*> Jcol_qf(dim);
   for (int k = 0; k < dim; k++)
   {
      Jcol_qf[k] = new QuadratureFunction(&qs, dim);
      *Jcol_qf[k] = 0.0;
   }

   DenseMatrix J(dim), Jinv(dim);
   Vector col(dim);

   // Flat iteration over quadrature points
   // MFEM QuadratureFunctions are stored as (NE * NQ) or (NE * NQ * vdim)
   // We iterate elements and local quad points to compute physics
   int offset = 0;
   
   for (int e = 0; e < pmesh.GetNE(); e++)
   {
      ElementTransformation *T = pmesh.GetElementTransformation(e);
      // We must use the integration rule from the QuadratureSpace to match indices
      const IntegrationRule &ir = qs.GetIntRule(e);

      for (int q = 0; q < ir.GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         T->SetIntPoint(&ip);

         // Get Jacobian
         J = T->Jacobian();
         const double detJ = J.Det();
         
         // 1. detJ
         detJ_qf[offset] = detJ;

         // 2. Frobenius norm of J
         double jf2 = 0.0;
         for (int i = 0; i < dim; i++)
            for (int k = 0; k < dim; k++)
               jf2 += J(i,k)*J(i,k);
         const double jf = std::sqrt(jf2);
         Jfrob_qf[offset] = jf;

         // 3. Columns of J as vectors
         // QuadratureFunction vector layout is usually byVDIM: [q1_x, q2_x, ... q1_y, q2_y ...]
         // But accessing via Vector view at index 'offset' handles strides if we use SetVectorValue?
         // No, standard QuadratureFunction doesn't have SetVectorValue.
         // We have to know the layout. By default, QuadratureFunction is standard Vector.
         // If vdim > 1, it stores values physically sequentially?
         // Actually, let's use the helper GetElementValues if possible, or manual.
         // Manual flat index: values are usually ordered by (element, point, vdim) or (vdim, element, point)
         // depending on ordering. QuadratureFunction default is typically (element, point) blocks.
         // Let's assume standard VDof ordering for QuadratureFunction is (NE*NQ) blocks.
         // Actually, VectorQuadratureFunction is not a standard class in older MFEM.
         // Let's assume we map the QuadratureFunction which is just a Vector.
         
         // For vector QF, index i is (e*NQ + q) + component * (NE*NQ) usually? 
         // Or interleaved?
         // Safer approach: Use the ParGridFunction projection which handles everything!
         // But we need to fill the QF first.
         
         // Let's assume standard ordering: data[ k * total_size + idx ]
         // Actually, let's look at how we constructed it: QuadratureFunction(&qs, dim)
         // The underlying storage is a Vector of size qs.GetSize() * dim.
         // Access: qf( index * vdim + component ) if ordering is byNODES?
         // No, QuadratureFunction doesn't support GetValue(element, ip).
         
         // Simple fix: Write directly to the data array if we know the layout.
         // MFEM defaults: Ordering::byNODES (VDIM outer) or byVDIM (VDIM inner).
         // QuadratureFunction uses byVDIM (interleaved) usually?
         // Let's use a simpler approach: 
         // We don't strictly *need* QuadratureFunctions for the vectors if we just want to project.
         // But ProjectCoefficient takes a Coefficient.
         
         // Let's assume interleaved for now: [x1, y1, z1, x2, y2, z2...] for each point.
         // int vec_offset = offset * dim;
         
         // Actually, let's check VectorQuadratureFunctionCoefficient. It expects a VectorQuadratureFunction.
         // Since that class doesn't exist, we will use VectorGridFunctionCoefficient if we had a grid function.
         
         // Alternative: Implement a VectorCoefficient that evaluates J on the fly.
         // This is MUCH cleaner and avoids QF layout guessing.
         // I will switch to that approach for the columns below.
         
         // 4. log(detJ)
         const double det_clamp = (detJ > det_eps) ? detJ : det_eps;
         logdetJ_qf[offset] = std::log(det_clamp);

         // 5. Inverse Frobenius
         double jinvf = 0.0;
         if (std::abs(detJ) > det_eps)
         {
            CalcInverse(J, Jinv);
            double jinvf2 = 0.0;
            for (int i = 0; i < dim; i++)
               for (int k = 0; k < dim; k++)
                  jinvf2 += Jinv(i,k)*Jinv(i,k);
            jinvf = std::sqrt(jinvf2);
         }
         else
         {
            jinvf = 1.0 / det_eps;
         }
         Jinvfrob_qf[offset] = jinvf;
         cond_qf[offset] = jf * jinvf;

         offset++;
      }
   }

   // Project scalars
   {
       QuadratureFunctionCoefficient c(detJ_qf);
       detJ_gf.ProjectCoefficient(c);
   }
   {
       QuadratureFunctionCoefficient c(logdetJ_qf);
       logdetJ_gf.ProjectCoefficient(c);
   }
   {
       QuadratureFunctionCoefficient c(Jfrob_qf);
       Jfrob_gf.ProjectCoefficient(c);
   }
   {
       QuadratureFunctionCoefficient c(Jinvfrob_qf);
       Jinvfrob_gf.ProjectCoefficient(c);
   }
   {
       QuadratureFunctionCoefficient c(cond_qf);
       cond_gf.ProjectCoefficient(c);
   }

   // For columns, we define a small coefficient class locally
   class JacobianColumnCoefficient : public VectorCoefficient
   {
   private:
      int col_idx;
   public:
      JacobianColumnCoefficient(int dim, int col) 
         : VectorCoefficient(dim), col_idx(col) {}
         
      virtual void Eval(Vector &V, ElementTransformation &T, 
                        const IntegrationPoint &ip)
      {
         // Set the integration point first!
         T.SetIntPoint(&ip);
         // Get the Jacobian (MFEM returns const reference)
         const DenseMatrix &Jlocal = T.Jacobian();
         
         int d = Jlocal.Height();
         V.SetSize(d);
         for(int i=0; i<d; i++) V(i) = Jlocal(i, col_idx);
      }
   };

   for (int k = 0; k < dim; k++)
   {
      JacobianColumnCoefficient j_col_coeff(dim, k);
      Jcol_gf[k]->ProjectCoefficient(j_col_coeff);
      delete Jcol_qf[k];
   }
}

void ComputeViscousBaroclinic(ParGridFunction &rho_h1, ParGridFunction &accel_tau,
                              ParGridFunction &w_visc_baro)
{
   ParFiniteElementSpace *h1_fes = rho_h1.ParFESpace();
   ParFiniteElementSpace *wfes = w_visc_baro.ParFESpace();

   rho_h1.HostRead();
   accel_tau.HostRead();
   w_visc_baro.HostReadWrite();

   int dim = h1_fes->GetMesh()->Dimension();

   w_visc_baro = 0.0;

   Array<int> zones_per_vdof;
   zones_per_vdof.SetSize(wfes->GetVSize());
   zones_per_vdof = 0;

   Array<int> vdofs;
   Vector grad_rho;
   Vector a_val; 
   Vector baro_val;

   for (int e = 0; e < h1_fes->GetNE(); ++e)
   {
      wfes->GetElementVDofs(e, vdofs);
      ElementTransformation *tr = h1_fes->GetElementTransformation(e);
      const FiniteElement *el = h1_fes->GetFE(e);
      int elndofs = el->GetDof();

      for (int dof = 0; dof < elndofs; ++dof)
      {
         const IntegrationPoint &ip = el->GetNodes().IntPoint(dof);
         tr->SetIntPoint(&ip);

         rho_h1.GetGradient(*tr, grad_rho);
         accel_tau.GetVectorValue(*tr, ip, a_val);

         double rho_val = rho_h1.GetValue(*tr);
         // Term is - (1/rho) * (grad_rho x a_visc)
         double scale = -1.0 / (rho_val + 1e-12);

         if (dim == 2)
         {
             // 2D Cross product (scalar result)
             double cross_z = grad_rho(0) * a_val(1) - grad_rho(1) * a_val(0);
             w_visc_baro(vdofs[dof]) += scale * cross_z;
             zones_per_vdof[vdofs[dof]]++;
         }
         else
         {
             baro_val.SetSize(3);
             baro_val(0) = grad_rho(1)*a_val(2) - grad_rho(2)*a_val(1);
             baro_val(1) = grad_rho(2)*a_val(0) - grad_rho(0)*a_val(2);
             baro_val(2) = grad_rho(0)*a_val(1) - grad_rho(1)*a_val(0);

             baro_val *= scale;

             bool by_nodes = (wfes->GetOrdering() == Ordering::byNODES);
             int vdim_local = wfes->GetVDim();

             for (int j = 0; j < vdim_local; ++j)
             {
                 int ldof = by_nodes ? vdofs[j*elndofs + dof] : vdofs[dof*vdim_local + j];
                 w_visc_baro(ldof) += baro_val(j);
                 zones_per_vdof[ldof]++;
             }
         }
      }
   }

   GroupCommunicator &gcomm = wfes->GroupComm();
   gcomm.Reduce<int>(zones_per_vdof, GroupCommunicator::Sum);
   gcomm.Bcast<int>(zones_per_vdof);

   gcomm.Reduce<double>(w_visc_baro.GetData(), GroupCommunicator::Sum);
   gcomm.Bcast<double>(w_visc_baro.GetData());

   for (int i = 0; i < zones_per_vdof.Size(); i++)
   {
      if (zones_per_vdof[i]) w_visc_baro(i) /= zones_per_vdof[i];
   }
}

void ComputeViscousTorqueComponents(ParGridFunction &rho_h1, ParGridFunction &accel_tau,
                                    ParGridFunction &w_visc_total,
                                    ParGridFunction &w_visc_diff,
                                    ParGridFunction &w_visc_baro)
{
   ComputeCurl(accel_tau, w_visc_total);
   ComputeViscousBaroclinic(rho_h1, accel_tau, w_visc_baro);

   // Diffusion = Total - Baroclinic
   w_visc_diff = w_visc_total;
   w_visc_diff.Add(-1.0, w_visc_baro);
}

} // namespace Diagnostics

int main(int argc, char *argv[])
{
   // Initialize MPI.
   Mpi::Init();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // Print the banner.
   if (Mpi::Root()) { display_banner(cout); }

   // Parse command-line options.
   problem = 1;
   dim = 3;
   const char *mesh_file = "default";
   int rs_levels = 2;
   int rp_levels = 0;
   Array<int> cxyz;
   int order_v = 2;
   int order_e = 1;
   int order_q = -1;
   int ode_solver_type = 4;
   double t_final = 0.6;
   double fixed_dt = -1.0;
   double cfl = 0.5;
   double cg_tol = 1e-8;
   double ftz_tol = 0.0;
   int cg_max_iter = 300;
   int max_tsteps = -1;
   int interp_cycle = 0;
   double interp_dt = 0.0;
   const char *interp_mesh_dir = "interp_mesh";
   bool p_assembly = true;
   bool impose_visc = false;
   bool fixed_viscosity = false;
   double reynolds = -1.0;
   double viscosity_const = -1.0;
   double kappa_e = 0.0;
   bool use_conduction = false;
   double prandtl_number = 0.71;
   bool visualization = false;
   int vis_steps = 5;
   bool visit = false;
   bool gfprint = false;
   const char *basename = "results/Laghos";
   const char *diag_file = "";
   int partition_type = 0;
   const char *device = "cpu";
   bool check = false;
   bool mem_usage = false;
   bool fom = false;
   bool gpu_aware_mpi = false;
   int dev = 0;
   double blast_energy = 0.25;
   double blast_position[] = {0.0, 0.0, 0.0};

   bool enable_nc = true;
   bool enable_rebalance = true;

   OptionsParser args(argc, argv);
   args.AddOption(&dim, "-dim", "--dimension", "Dimension of the problem.");
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&rp_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&cxyz, "-c", "--cartesian-partitioning",
                  "Use Cartesian partitioning.");
   args.AddOption(&problem, "-p", "--problem", "Problem setup to use.");
   args.AddOption(&order_v, "-ok", "--order-kinematic",
                  "Order (degree) of the kinematic finite element space.");
   args.AddOption(&order_e, "-ot", "--order-thermo",
                  "Order (degree) of the thermodynamic finite element space.");
   args.AddOption(&order_q, "-oq", "--order-intrule",
                  "Order  of the integration rule.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 1 - Forward Euler,\n\t"
                  "            2 - RK2 SSP, 3 - RK3 SSP, 4 - RK4, 6 - RK6,\n\t"
                  "            7 - RK2Avg.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&fixed_dt, "-dt", "--time-step",
                  "Fixed time step (disables CFL-based adaptivity).");
   args.AddOption(&cfl, "-cfl", "--cfl", "CFL-condition number.");
   args.AddOption(&cg_tol, "-cgt", "--cg-tol",
                  "Relative CG tolerance (velocity linear solve).");
   args.AddOption(&ftz_tol, "-ftz", "--ftz-tol",
                  "Absolute flush-to-zero tolerance.");
   args.AddOption(&cg_max_iter, "-cgm", "--cg-max-steps",
                  "Maximum number of CG iterations (velocity linear solve).");
   args.AddOption(&max_tsteps, "-ms", "--max-steps",
                  "Maximum number of steps (negative means no restriction).");
   args.AddOption(&interp_cycle, "-ic", "--interp-cycle",
                  "Remap every N cycles (0 disables).");
   args.AddOption(&interp_dt, "-it", "--interp-time",
                  "Remap every time interval (0 disables).");
   args.AddOption(&interp_mesh_dir, "-imd", "--interp-mesh-dir",
                  "Directory for saving the initial mesh partitions.");
   args.AddOption(&p_assembly, "-pa", "--partial-assembly", "-fa",
                  "--full-assembly",
                  "Activate 1D tensor-based assembly (partial assembly).");
   args.AddOption(&impose_visc, "-iv", "--impose-viscosity", "-niv",
                  "--no-impose-viscosity",
                  "Use active viscosity terms even for smooth problems.");
   args.AddOption(&fixed_viscosity, "-fv", "--fixed-viscosity", "-nfv",
                  "--no-fixed-viscosity",
                  "Use constant physical viscosity (overrides artificial).");
   args.AddOption(&reynolds, "-Re", "--reynolds",
                  "Reynolds number for fixed viscosity (L=1/(2*pi)).");
   args.AddOption(&use_conduction, "-cond", "--conduction", "-no-cond",
                  "--no-conduction",
                  "Enable or disable heat conduction.");
   args.AddOption(&prandtl_number, "-pr", "--prandtl",
                  "Prandtl number for heat conduction.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.AddOption(&visit, "-visit", "--visit", "-no-visit", "--no-visit",
                  "Enable or disable VisIt visualization.");
   args.AddOption(&gfprint, "-print", "--print", "-no-print", "--no-print",
                  "Enable or disable result output (files in mfem format).");
   args.AddOption(&basename, "-k", "--outputfilename",
                  "Name of the visit dump files");
   args.AddOption(&p0_user, "-p0", "--pressure0",
                  "Background pressure for problem 0 (used if -mach not set).");
   args.AddOption(&mach_number, "-mach", "--mach-number",
                  "Mach number for problem 0 (overrides -p0).");
   args.AddOption(&mach_u0, "-u0", "--mach-uref",
                  "Reference velocity for Mach number in problem 0.");
   args.AddOption(&diag_file, "-diag", "--diag-file",
                  "Write per-step diagnostics to a text file (root rank).");
   args.AddOption(&partition_type, "-pt", "--partition",
                  "Customized x/y/z Cartesian MPI partitioning of the serial mesh.\n\t"
                  "Here x,y,z are relative task ratios in each direction.\n\t"
                  "Example: with 48 mpi tasks and -pt 321, one would get a Cartesian\n\t"
                  "partition of the serial mesh by (6,4,2) MPI tasks in (x,y,z).\n\t"
                  "NOTE: the serially refined mesh must have the appropriate number\n\t"
                  "of zones in each direction, e.g., the number of zones in direction x\n\t"
                  "must be divisible by the number of MPI tasks in direction x.\n\t"
                  "Available options: 11, 21, 111, 211, 221, 311, 321, 322, 432.");
   args.AddOption(&device, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&check, "-chk", "--checks", "-no-chk", "--no-checks",
                  "Enable 2D checks.");
   args.AddOption(&mem_usage, "-mb", "--mem", "-no-mem", "--no-mem",
                  "Enable memory usage.");
   args.AddOption(&fom, "-f", "--fom", "-no-fom", "--no-fom",
                  "Enable figure of merit output.");
   args.AddOption(&gpu_aware_mpi, "-gam", "--gpu-aware-mpi", "-no-gam",
                  "--no-gpu-aware-mpi", "Enable GPU aware MPI communications.");
   args.AddOption(&enable_nc, "-nc", "--nonconforming", "-no-nc",
                  "--conforming",
                  "Use non-conforming meshes. Requires a 2D or 3D mesh.");
   args.AddOption(&enable_rebalance, "-b", "--balance", "-no-b",
                  "--no-rebalance",
                  "Perform a rebalance after parallel refinement. Only enabled \n\t"
                  "for non-conforming meshes with Metis partitioning.");
   args.AddOption(&dev, "-dev", "--dev", "GPU device to use.");
   args.Parse();
   if (!args.Good())
   {
      if (Mpi::Root()) { args.PrintUsage(cout); }
      return 1;
   }
   if (Mpi::Root()) { args.PrintOptions(cout); }

   const bool interp_cycle_on = interp_cycle > 0;
   const bool interp_time_on = interp_dt > 0.0;
   if (interp_cycle < 0 || interp_dt < 0.0)
   {
      if (Mpi::Root())
      {
         cerr << "Interpolation options must be non-negative." << endl;
      }
      return 1;
   }
   if (interp_cycle_on && interp_time_on)
   {
      if (Mpi::Root())
      {
         cerr << "Choose only one interpolation trigger: "
                 "--interp-cycle or --interp-time." << endl;
      }
      return 1;
   }
   const bool interp_enabled = interp_cycle_on || interp_time_on;
#ifndef MFEM_USE_GSLIB
   if (interp_enabled)
   {
      if (Mpi::Root())
      {
         cerr << "Interpolation requires MFEM built with GSLIB "
                 "(MFEM_USE_GSLIB=YES)." << endl;
      }
      return 1;
   }
#endif

   // Configure the device from the command line options
   Device backend;
   backend.Configure(device, dev);
   if (Mpi::Root()) { backend.Print(); }
   backend.SetGPUAwareMPI(gpu_aware_mpi);

   // On all processors, use the default builtin 1D/2D/3D mesh or read the
   // serial one given on the command line.
   Mesh *mesh;
   if (strncmp(mesh_file, "default", 7) != 0)
   {
      mesh = new Mesh(mesh_file, true, true);
   }
   else
   {
      if (dim == 1)
      {
         mesh = new Mesh(Mesh::MakeCartesian1D(2));
         mesh->GetBdrElement(0)->SetAttribute(1);
         mesh->GetBdrElement(1)->SetAttribute(1);
      }
      if (dim == 2)
      {
         mesh = new Mesh(Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL,
                                               true));
         const int NBE = mesh->GetNBE();
         for (int b = 0; b < NBE; b++)
         {
            Element *bel = mesh->GetBdrElement(b);
            const int attr = (b < NBE/2) ? 2 : 1;
            bel->SetAttribute(attr);
         }
      }
      if (dim == 3)
      {
         mesh = new Mesh(Mesh::MakeCartesian3D(2, 2, 2, Element::HEXAHEDRON,
                                               true));
         const int NBE = mesh->GetNBE();
         for (int b = 0; b < NBE; b++)
         {
            Element *bel = mesh->GetBdrElement(b);
            const int attr = (b < NBE/3) ? 3 : (b < 2*NBE/3) ? 1 : 2;
            bel->SetAttribute(attr);
         }
      }
   }
   dim = mesh->Dimension();

   // 1D vs partial assembly sanity check.
   if (p_assembly && dim == 1)
   {
      p_assembly = false;
      if (Mpi::Root())
      {
         cout << "Laghos does not support PA in 1D. Switching to FA." << endl;
      }
   }

   if (enable_nc && dim > 1)
   {
      if (Mpi::Root())
      {
         cout << "Using non-conforming mesh." << endl;
      }
      mesh->EnsureNCMesh();
   }

   // Refine the mesh in serial to increase the resolution.
   for (int lev = 0; lev < rs_levels; lev++) { mesh->UniformRefinement(); }
   const int mesh_NE = mesh->GetNE();
   if (Mpi::Root())
   {
      cout << "Number of zones in the serial mesh: " << mesh_NE << endl;
   }

   // Parallel partitioning of the mesh.
   ParMesh *pmesh = nullptr;
   const int num_tasks = Mpi::WorldSize(); int unit = 1;
   int *nxyz = new int[dim];
   switch (partition_type)
   {
      case 0:
         for (int d = 0; d < dim; d++) { nxyz[d] = unit; }
         break;
      case 11:
      case 111:
         unit = static_cast<int>(floor(pow(num_tasks, 1.0 / dim) + 1e-2));
         for (int d = 0; d < dim; d++) { nxyz[d] = unit; }
         break;
      case 21: // 2D
         unit = static_cast<int>(floor(pow(num_tasks / 2, 1.0 / 2) + 1e-2));
         nxyz[0] = 2 * unit; nxyz[1] = unit;
         break;
      case 31: // 2D
         unit = static_cast<int>(floor(pow(num_tasks / 3, 1.0 / 2) + 1e-2));
         nxyz[0] = 3 * unit; nxyz[1] = unit;
         break;
      case 32: // 2D
         unit = static_cast<int>(floor(pow(2 * num_tasks / 3, 1.0 / 2) + 1e-2));
         nxyz[0] = 3 * unit / 2; nxyz[1] = unit;
         break;
      case 49: // 2D
         unit = static_cast<int>(floor(pow(9 * num_tasks / 4, 1.0 / 2) + 1e-2));
         nxyz[0] = 4 * unit / 9; nxyz[1] = unit;
         break;
      case 51: // 2D
         unit = static_cast<int>(floor(pow(num_tasks / 5, 1.0 / 2) + 1e-2));
         nxyz[0] = 5 * unit; nxyz[1] = unit;
         break;
      case 211: // 3D.
         unit = static_cast<int>(floor(pow(num_tasks / 2, 1.0 / 3) + 1e-2));
         nxyz[0] = 2 * unit; nxyz[1] = unit; nxyz[2] = unit;
         break;
      case 221: // 3D.
         unit = static_cast<int>(floor(pow(num_tasks / 4, 1.0 / 3) + 1e-2));
         nxyz[0] = 2 * unit; nxyz[1] = 2 * unit; nxyz[2] = unit;
         break;
      case 311: // 3D.
         unit = static_cast<int>(floor(pow(num_tasks / 3, 1.0 / 3) + 1e-2));
         nxyz[0] = 3 * unit; nxyz[1] = unit; nxyz[2] = unit;
         break;
      case 321: // 3D.
         unit = static_cast<int>(floor(pow(num_tasks / 6, 1.0 / 3) + 1e-2));
         nxyz[0] = 3 * unit; nxyz[1] = 2 * unit; nxyz[2] = unit;
         break;
      case 322: // 3D.
         unit = static_cast<int>(floor(pow(2 * num_tasks / 3, 1.0 / 3) + 1e-2));
         nxyz[0] = 3 * unit / 2; nxyz[1] = unit; nxyz[2] = unit;
         break;
      case 432: // 3D.
         unit = static_cast<int>(floor(pow(num_tasks / 3, 1.0 / 3) + 1e-2));
         nxyz[0] = 2 * unit; nxyz[1] = 3 * unit / 2; nxyz[2] = unit;
         break;
      case 511: // 3D.
         unit = static_cast<int>(floor(pow(num_tasks / 5, 1.0 / 3) + 1e-2));
         nxyz[0] = 5 * unit; nxyz[1] = unit; nxyz[2] = unit;
         break;
      case 521: // 3D.
         unit = static_cast<int>(floor(pow(num_tasks / 10, 1.0 / 3) + 1e-2));
         nxyz[0] = 5 * unit; nxyz[1] = 2 * unit; nxyz[2] = unit;
         break;
      case 522: // 3D.
         unit = static_cast<int>(floor(pow(num_tasks / 20, 1.0 / 3) + 1e-2));
         nxyz[0] = 5 * unit; nxyz[1] = 2 * unit; nxyz[2] = 2 * unit;
         break;
      case 911: // 3D.
         unit = static_cast<int>(floor(pow(num_tasks / 9, 1.0 / 3) + 1e-2));
         nxyz[0] = 9 * unit; nxyz[1] = unit; nxyz[2] = unit;
         break;
      case 921: // 3D.
         unit = static_cast<int>(floor(pow(num_tasks / 18, 1.0 / 3) + 1e-2));
         nxyz[0] = 9 * unit; nxyz[1] = 2 * unit; nxyz[2] = unit;
         break;
      case 922: // 3D.
         unit = static_cast<int>(floor(pow(num_tasks / 36, 1.0 / 3) + 1e-2));
         nxyz[0] = 9 * unit; nxyz[1] = 2 * unit; nxyz[2] = 2 * unit;
         break;
      default:
         if (myid == 0)
         {
            cout << "Unknown partition type: " << partition_type << '\n';
         }
         delete mesh;
         MPI_Finalize();
         return 3;
   }
   int product = 1;
   for (int d = 0; d < dim; d++) { product *= nxyz[d]; }
   const bool cartesian_partitioning = (cxyz.Size()>0)?true:false;
   if (product == num_tasks || cartesian_partitioning)
   {
      if (cartesian_partitioning)
      {
         int cproduct = 1;
         for (int d = 0; d < dim; d++) { cproduct *= cxyz[d]; }
         MFEM_VERIFY(!cartesian_partitioning || cxyz.Size() == dim,
                     "Expected " << mesh->SpaceDimension() << " integers with the "
                     "option --cartesian-partitioning.");
         MFEM_VERIFY(!cartesian_partitioning || num_tasks == cproduct,
                     "Expected cartesian partitioning product to match number of ranks.");
      }
      int *partitioning = cartesian_partitioning ?
                          mesh->CartesianPartitioning(cxyz):
                          mesh->CartesianPartitioning(nxyz);
      pmesh = new ParMesh(MPI_COMM_WORLD, *mesh, partitioning);
      delete [] partitioning;
   }
   else
   {
      if (myid == 0)
      {
         cout << "Non-Cartesian partitioning through METIS will be used.\n";
#ifndef MFEM_USE_METIS
         cout << "MFEM was built without METIS. "
              << "Adjust the number of tasks to use a Cartesian split." << endl;
#endif
      }
#ifndef MFEM_USE_METIS
      return 1;
#endif
      pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   }
   delete [] nxyz;
   delete mesh;

   // Refine the mesh further in parallel to increase the resolution.
   for (int lev = 0; lev < rp_levels; lev++) { pmesh->UniformRefinement(); }

   pmesh->GetBoundingBox(bb_min, bb_max);

   if (!cartesian_partitioning && enable_nc && dim > 1)
   {
      if (myid == 0) { cout << "Rebalancing mesh" << endl; }
      pmesh->Rebalance();
   }

   int NE = pmesh->GetNE(), ne_min, ne_max;
   MPI_Reduce(&NE, &ne_min, 1, MPI_INT, MPI_MIN, 0, pmesh->GetComm());
   MPI_Reduce(&NE, &ne_max, 1, MPI_INT, MPI_MAX, 0, pmesh->GetComm());
   if (myid == 0)
   { cout << "Zones min/max: " << ne_min << " " << ne_max << endl; }

   // Define the parallel finite element spaces. We use:
   // - H1 (Gauss-Lobatto, continuous) for position and velocity.
   // - L2 (Bernstein, discontinuous) for specific internal energy.
   L2_FECollection L2FEC(order_e, dim, BasisType::Positive);
   H1_FECollection H1FEC(order_v, dim);
   ParFiniteElementSpace L2FESpace(pmesh, &L2FEC);
   ParFiniteElementSpace H1FESpace(pmesh, &H1FEC, pmesh->Dimension());
   ParFiniteElementSpace H1ScalarFESpace(pmesh, &H1FEC); // For 2D vorticity

   // Boundary conditions: all tests use v.n = 0 on the boundary, and we assume
   // that the boundaries are straight.
   Array<int> ess_tdofs, ess_vdofs;
   {
      Array<int> ess_bdr(pmesh->bdr_attributes.Max()), dofs_marker, dofs_list;
      for (int d = 0; d < pmesh->Dimension(); d++)
      {
         // Attributes 1/2/3 correspond to fixed-x/y/z boundaries,
         // i.e., we must enforce v_x/y/z = 0 for the velocity components.
         ess_bdr = 0; ess_bdr[d] = 1;
         H1FESpace.GetEssentialTrueDofs(ess_bdr, dofs_list, d);
         ess_tdofs.Append(dofs_list);
         H1FESpace.GetEssentialVDofs(ess_bdr, dofs_marker, d);
         FiniteElementSpace::MarkerToList(dofs_marker, dofs_list);
         ess_vdofs.Append(dofs_list);
      }
   }

   // Define the explicit ODE solver used for time integration.
   ODESolver *ode_solver = NULL;
   switch (ode_solver_type)
   {
      case 1: ode_solver = new ForwardEulerSolver; break;
      case 2: ode_solver = new RK2Solver(0.5); break;
      case 3: ode_solver = new RK3SSPSolver; break;
      case 4: ode_solver = new RK4Solver; break;
      case 6: ode_solver = new RK6Solver; break;
      case 7: ode_solver = new RK2AvgSolver; break;
      default:
         if (myid == 0)
         {
            cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
         }
         delete pmesh;
         MPI_Finalize();
         return 3;
   }

   const HYPRE_BigInt glob_size_l2 = L2FESpace.GlobalTrueVSize();
   const HYPRE_BigInt glob_size_h1 = H1FESpace.GlobalTrueVSize();
   if (Mpi::Root())
   {
      cout << "Number of kinematic (position, velocity) dofs: "
           << glob_size_h1 << endl;
      cout << "Number of specific internal energy dofs: "
           << glob_size_l2 << endl;
   }

   // The monolithic BlockVector stores unknown fields as:
   // - 0 -> position
   // - 1 -> velocity
   // - 2 -> specific internal energy
   const int Vsize_l2 = L2FESpace.GetVSize();
   const int Vsize_h1 = H1FESpace.GetVSize();
   Array<int> offset(4);
   offset[0] = 0;
   offset[1] = offset[0] + Vsize_h1;
   offset[2] = offset[1] + Vsize_h1;
   offset[3] = offset[2] + Vsize_l2;
   BlockVector S(offset, Device::GetMemoryType());

   // Define GridFunction objects for the position, velocity and specific
   // internal energy. There is no function for the density, as we can always
   // compute the density values given the current mesh position, using the
   // property of pointwise mass conservation.
   ParGridFunction x_gf, v_gf, e_gf;
   ParGridFunction w_gf; // Vorticity
   ParGridFunction w_stretch, w_compress; // Vorticity Budget Terms
   ParGridFunction w_baroclinic; // Baroclinic Torque
   ParGridFunction w_viscous; // Total Viscous Torque (Curl of a_visc)
   ParGridFunction w_visc_diff, w_visc_baro; // Viscous Diffusion & Viscous Baroclinic
   ParGridFunction work_p, work_tau, work_total; // Work Diagnostic Fields
   ParGridFunction accel_gf; // Acceleration Field (RHS of Momentum)
   ParGridFunction accel_p, accel_tau; // Components of Acceleration
   ParGridFunction rho_h1, e_h1; // Intermediate H1 fields for baroclinic term
   ParGridFunction v_visc_gf; // Intermediate H1 field for viscous acceleration

   ParGridFunction x0_gf(&H1FESpace);
   x_gf.MakeRef(&H1FESpace, S, offset[0]);
   v_gf.MakeRef(&H1FESpace, S, offset[1]);
   e_gf.MakeRef(&L2FESpace, S, offset[2]);

   // Initialize vorticity grid function
   if (dim == 2)
   {
      w_gf.SetSpace(&H1ScalarFESpace);
      w_stretch.SetSpace(&H1ScalarFESpace);
      w_compress.SetSpace(&H1ScalarFESpace);
      w_baroclinic.SetSpace(&H1ScalarFESpace);
      w_viscous.SetSpace(&H1ScalarFESpace);
      w_visc_diff.SetSpace(&H1ScalarFESpace);
      w_visc_baro.SetSpace(&H1ScalarFESpace);
   }
   else
   {
      w_gf.SetSpace(&H1FESpace);
      w_stretch.SetSpace(&H1FESpace);
      w_compress.SetSpace(&H1FESpace);
      w_baroclinic.SetSpace(&H1FESpace);
      w_viscous.SetSpace(&H1FESpace);
      w_visc_diff.SetSpace(&H1FESpace);
      w_visc_baro.SetSpace(&H1FESpace);
   }
   w_gf = 0.0;
   w_stretch = 0.0;
   w_compress = 0.0;
   w_baroclinic = 0.0;
   w_viscous = 0.0;
   w_visc_diff = 0.0;
   w_visc_baro = 0.0;

   work_p.SetSpace(&L2FESpace);
   work_tau.SetSpace(&L2FESpace);
   work_total.SetSpace(&L2FESpace);
   work_p = 0.0;
   work_tau = 0.0;
   work_total = 0.0;

   accel_gf.SetSpace(&H1FESpace);
   accel_gf = 0.0;
   accel_p.SetSpace(&H1FESpace);
   accel_p = 0.0;
   accel_tau.SetSpace(&H1FESpace);
   accel_tau = 0.0;

   rho_h1.SetSpace(&H1ScalarFESpace);
   e_h1.SetSpace(&H1ScalarFESpace);
   rho_h1 = 0.0;
   e_h1 = 0.0;
   v_visc_gf.SetSpace(&H1FESpace);
   v_visc_gf = 0.0;

   ParGridFunction T_gf(&L2FESpace);
   T_gf = 0.0;

   // Integral of specific internal energy for problem 8 (heat conduction compare)
   ConstantCoefficient one_e(1.0);
   ParLinearForm e_int_lf(&L2FESpace);
   const bool track_e_integral = (problem == 8);
   if (track_e_integral)
   {
      e_int_lf.AddDomainIntegrator(new DomainLFIntegrator(one_e));
      e_int_lf.Assemble();
   }

   // ---- Jacobian diagnostics fields ----
   const int vis_order = order_v;
   const int qorder    = 2*order_v; 

   mfem::L2_FECollection vis_l2_fec(vis_order, dim);
   mfem::ParFiniteElementSpace l2s_fes(pmesh, &vis_l2_fec);           // scalar
   mfem::ParFiniteElementSpace l2v_fes(pmesh, &vis_l2_fec, dim);      // vector (vdim=dim)

   // Scalars
   mfem::ParGridFunction detJ_gf(&l2s_fes);
   mfem::ParGridFunction logdetJ_gf(&l2s_fes);
   mfem::ParGridFunction Jfrob_gf(&l2s_fes);
   mfem::ParGridFunction Jinvfrob_gf(&l2s_fes);
   mfem::ParGridFunction cond_gf(&l2s_fes);

   // Vectors: J columns
   std::vector<mfem::ParGridFunction*> Jcol_gf(dim);
   for (int k = 0; k < dim; k++) { Jcol_gf[k] = new mfem::ParGridFunction(&l2v_fes); }

   // init
   detJ_gf = 0.0; logdetJ_gf = 0.0; Jfrob_gf = 0.0; Jinvfrob_gf = 0.0; cond_gf = 0.0;
   for (int k = 0; k < dim; k++) { (*Jcol_gf[k]) = 0.0; }

   // Initialize x_gf using the starting mesh coordinates.
   pmesh->SetNodalGridFunction(&x_gf);
   // Sync the data location of x_gf with its base, S
   x_gf.SyncAliasMemory(S);
   x0_gf = x_gf;
   if (interp_enabled)
   {
      SaveParallelMesh(*pmesh, interp_mesh_dir, "initial_mesh");
   }

   // Initialize the velocity.
   VectorFunctionCoefficient v_coeff(pmesh->Dimension(), v0);
   v_gf.ProjectCoefficient(v_coeff);
   for (int i = 0; i < ess_vdofs.Size(); i++)
   {
      v_gf(ess_vdofs[i]) = 0.0;
   }
   // Sync the data location of v_gf with its base, S
   v_gf.SyncAliasMemory(S);

   // Initialize density and specific internal energy values. We interpolate in
   // a non-positive basis to get the correct values at the dofs. Then we do an
   // L2 projection to the positive basis in which we actually compute. The goal
   // is to get a high-order representation of the initial condition. Note that
   // this density is a temporary function and it will not be updated during the
   // time evolution.
   if (problem == 0)
   {
      Vector x0(dim); x0 = 0.0;
      const double rho_ref = rho0(x0);
      const double gamma_ref = gamma_func(x0);
      double mach_eff = mach_number;
      if (mach_eff <= 0.0)
      {
         const double p0_ref = (dim == 3) ? 100.0 : 1.0;
         const double p0_input = (p0_user > 0.0) ? p0_user : p0_ref;
         const double c0 = sqrt(gamma_ref * p0_input / rho_ref);
         mach_eff = mach_u0 / c0;
      }
      p0_background = rho_ref * mach_u0 * mach_u0 /
                      (gamma_ref * mach_eff * mach_eff);
      if (Mpi::Root())
      {
         const double cs = sqrt(gamma_ref * p0_background / rho_ref);
         const double mach_out = mach_u0 / cs;
         const double L = 1.0 / (2.0 * M_PI);
         const double Eu0 = rho_ref * mach_u0 * mach_u0 *
                            (M_PI * L) * (M_PI * L) * (M_PI * L);
         cout << "The initial Mach number is: " << mach_out << endl;
         cout << "The background pressure is: " << p0_background << endl;
         cout << "The hydrodynamic energy is : " << Eu0 << endl;
      }
   }

   // Diagnostic for initial conduction state
   if (use_conduction && Mpi::Root())
   {
      cout << "Problem 8 Bounding Box: min = (" << bb_min(0) << "," << bb_min(1) << (dim==3 ? ","+std::to_string(bb_min(2)) : "") << "), "
           << "max = (" << bb_max(0) << "," << bb_max(1) << (dim==3 ? ","+std::to_string(bb_max(2)) : "") << ")" << endl;
   }
   ParGridFunction rho0_gf(&L2FESpace);
   FunctionCoefficient rho0_func_coeff(rho0);
   L2_FECollection l2_fec(order_e, pmesh->Dimension());
   ParFiniteElementSpace l2_fes(pmesh, &l2_fec);
   ParGridFunction l2_rho0_gf(&l2_fes), l2_e(&l2_fes);
   l2_rho0_gf.ProjectCoefficient(rho0_func_coeff);
   rho0_gf.ProjectGridFunction(l2_rho0_gf);
   if (problem == 1)
   {
      // For the Sedov test, we use a delta function at the origin.
      DeltaCoefficient e_coeff(blast_position[0], blast_position[1],
                               blast_position[2], blast_energy);
      l2_e.ProjectCoefficient(e_coeff);
   }
   else
   {
      FunctionCoefficient e_coeff(e0);
      l2_e.ProjectCoefficient(e_coeff);
   }
   e_gf.ProjectGridFunction(l2_e);
   if (problem == 8 && Mpi::Root())
   {
      cout << "e min/max: " << e_gf.Min() << " " << e_gf.Max() << endl;
   }
   // Sync the data location of e_gf with its base, S
   e_gf.SyncAliasMemory(S);

   if (problem == 8)
   {
      double loc_energy = e_gf * e_gf;
      double energy_init_dg;
      MPI_Allreduce(&loc_energy, &energy_init_dg, 1, MPI_DOUBLE, MPI_SUM, pmesh->GetComm());
      
      ParLinearForm LF_dg(&L2FESpace);
      ConstantCoefficient one_dg(1.0);
      LF_dg.AddDomainIntegrator(new DomainLFIntegrator(one_dg));
      LF_dg.Assemble();
      double loc_integral = LF_dg(e_gf);
      double integral_init_dg;
      MPI_Allreduce(&loc_integral, &integral_init_dg, 1, MPI_DOUBLE, MPI_SUM, pmesh->GetComm());

      if (Mpi::Root())
      {
         cout << "Initial L2 energy (e*e): " << energy_init_dg << endl;
         cout << "Initial total integral (e): " << integral_init_dg << endl;
      }
   }

   GridFunctionCoefficient rho0_coeff(&rho0_gf);

   // Piecewise constant ideal gas coefficient over the Lagrangian mesh. The
   // gamma values are projected on function that's constant on the moving mesh.
   L2_FECollection mat_fec(0, pmesh->Dimension());
   ParFiniteElementSpace mat_fes(pmesh, &mat_fec);
   ParGridFunction mat_gf(&mat_fes);
   FunctionCoefficient mat_coeff(gamma_func);
   mat_gf.ProjectCoefficient(mat_coeff);

   // Additional details, depending on the problem.
   int source = 0; bool visc = true, vorticity = false;
   switch (problem)
   {
      case 0: if (pmesh->Dimension() == 2) { source = 1; } visc = false; break;
      case 1: visc = true; break;
      case 2: visc = true; break;
      case 3: visc = true; S.HostRead(); break;
      case 4: visc = false; break;
      case 5: visc = true; break;
      case 6: visc = true; break;
      case 7: source = 2; visc = true; vorticity = true;  break;
      case 8: visc = true; break;
      default: MFEM_ABORT("Wrong problem specification!");
   }
   if (impose_visc) { visc = true; }
   if (fixed_viscosity)
   {
      MFEM_VERIFY(reynolds > 0.0,
                  "Fixed viscosity requires a positive Reynolds number.");
      Vector x0(dim); x0 = 0.0;
      const double rho_ref = rho0(x0);
      const double L = 1.0 / (2.0 * M_PI);
      viscosity_const = rho_ref * mach_u0 * L / reynolds;
      const double gamma_ref = gamma_func(x0);
      kappa_e = (prandtl_number > 0.0) ? (viscosity_const * gamma_ref) / prandtl_number : 0.0;
      if (Mpi::Root())
      {
         cout << "Fixed viscosity enabled: Re = " << reynolds
              << ", mu = " << viscosity_const << endl;
         if (use_conduction)
         {
            cout << "Heat conduction enabled: Pr = " << prandtl_number
                 << ", kappa = " << kappa_e
                 << endl;
         }
      }
      visc = true;
   }
   else
   {
      viscosity_const = -1.0;
   }

   if (use_conduction)
   {
      MFEM_VERIFY(fixed_viscosity,
                  "Heat conduction currently requires fixed viscosity (-fv).");
   }

   const bool freeze_momentum = (problem == 8);
   auto hydro = std::make_unique<hydrodynamics::LagrangianHydroOperator>(
      S.Size(),
      H1FESpace, L2FESpace, ess_tdofs,
      rho0_coeff, rho0_gf,
      mat_gf, source, cfl,
      visc, vorticity, viscosity_const,
      use_conduction, prandtl_number,
      freeze_momentum,
      p_assembly,
      cg_tol, cg_max_iter, ftz_tol,
      order_q);

   socketstream vis_rho, vis_v, vis_e;
   char vishost[] = "localhost";
   int  visport   = 19916;

   ParGridFunction rho_gf;
   if (visualization || visit) { hydro->ComputeDensity(rho_gf); }
   const double energy_init = hydro->InternalEnergy(e_gf) +
                              hydro->KineticEnergy(v_gf);

   if (visualization)
   {
      // Make sure all MPI ranks have sent their 'v' solution before initiating
      // another set of GLVis connections (one from each rank):
      MPI_Barrier(pmesh->GetComm());
      vis_rho.precision(8);
      vis_v.precision(8);
      vis_e.precision(8);
      int Wx = 0, Wy = 0; // window position
      const int Ww = 350, Wh = 350; // window size
      int offx = Ww+10; // window offsets
      if (problem != 0 && problem != 4)
      {
         hydrodynamics::VisualizeField(vis_rho, vishost, visport, rho_gf,
                                       "Density", Wx, Wy, Ww, Wh);
      }
      Wx += offx;
      hydrodynamics::VisualizeField(vis_v, vishost, visport, v_gf,
                                    "Velocity", Wx, Wy, Ww, Wh);
      Wx += offx;
      hydrodynamics::VisualizeField(vis_e, vishost, visport, e_gf,
                                    "Specific Internal Energy", Wx, Wy, Ww, Wh);
   }

   // Save data for VisIt visualization.
   VisItDataCollection visit_dc(basename, pmesh);
   VisItDataCollection visit_dc_debug("results_debug/Laghos", pmesh);
   if (visit)
   {
      // Compute Vorticity for t=0
      Diagnostics::ComputeCurl(v_gf, w_gf);
      Diagnostics::ComputeVortexStretching(v_gf, w_gf, w_stretch);
      Diagnostics::ComputeVortexCompression(v_gf, w_gf, w_compress);

      Diagnostics::ProjectL2toH1(rho_gf, rho_h1);
      Diagnostics::ProjectL2toH1(e_gf, e_h1);
      Diagnostics::ComputeBaroclinicTerm(rho_h1, e_h1, w_baroclinic);
      hydro->ComputeWorkFields(v_gf, work_p, work_tau, work_total);
      hydro->ComputeAcceleration(accel_gf, &accel_p, &accel_tau);

      if (visc)
      {
         Diagnostics::ComputeViscousTorqueComponents(rho_h1, accel_tau,
                                                     w_viscous, w_visc_diff, w_visc_baro);
      }

      Diagnostics::ComputeJacobian(const_cast<ParMesh&>(*pmesh),
                                      detJ_gf, logdetJ_gf,
                                      Jfrob_gf, Jinvfrob_gf, cond_gf,
                                      Jcol_gf,
                                      qorder);

      // if (visc)
      // {
      //    hydro->ComputeViscousAcceleration(S, v_visc_gf);
      //    ComputeCurl(v_visc_gf, w_viscous);
      // }

      T_gf = e_gf;
      // T = (gamma - 1) * e
      if (NE > 0) { T_gf *= (gamma_func(Vector({0.0})) - 1.0); }

      visit_dc.RegisterField("Density",  &rho_gf);
      visit_dc.RegisterField("Velocity", &v_gf);
      visit_dc.RegisterField("Specific Internal Energy", &e_gf);
      visit_dc.RegisterField("Temperature", &T_gf);
      visit_dc.RegisterField("Vorticity", &w_gf);
      visit_dc.SetCycle(0);
      visit_dc.SetTime(0.0);
      visit_dc.Save();

      visit_dc_debug.RegisterField("Vorticity", &w_gf);
      visit_dc_debug.RegisterField("Temperature", &T_gf);
      visit_dc_debug.RegisterField("VortexStretching", &w_stretch);
      visit_dc_debug.RegisterField("VortexCompression", &w_compress);
      visit_dc_debug.RegisterField("BaroclinicTorque", &w_baroclinic);
      visit_dc_debug.RegisterField("WorkPressure", &work_p);
      visit_dc_debug.RegisterField("WorkViscous", &work_tau);
      visit_dc_debug.RegisterField("WorkTotal", &work_total);
      visit_dc_debug.RegisterField("Acceleration", &accel_gf);
      visit_dc_debug.RegisterField("AccelPressure", &accel_p);
      visit_dc_debug.RegisterField("AccelViscous", &accel_tau);
      
      if (visc)
      {
         visit_dc_debug.RegisterField("ViscousTorqueTotal", &w_viscous);
         visit_dc_debug.RegisterField("ViscousDiffusionVorticity", &w_visc_diff);
         visit_dc_debug.RegisterField("ViscousBaroclinic", &w_visc_baro);
      }

      visit_dc_debug.RegisterField("detJ",      &detJ_gf);
      visit_dc_debug.RegisterField("log_detJ",  &logdetJ_gf);
      visit_dc_debug.RegisterField("J_Frob",    &Jfrob_gf);
      visit_dc_debug.RegisterField("Jinv_Frob", &Jinvfrob_gf);
      visit_dc_debug.RegisterField("cond_Frob", &cond_gf);

      for (int k = 0; k < dim; k++)
      {
         visit_dc_debug.RegisterField(("J_col"+std::to_string(k)).c_str(), Jcol_gf[k]);
      }

      visit_dc_debug.SetCycle(0);
      visit_dc_debug.SetTime(0.0);
      visit_dc_debug.Save();
   }
   std::ofstream diag_ofs;
   const bool diag_output = (diag_file && diag_file[0] != '\0');
   if (diag_output && Mpi::Root())
   {
      diag_ofs.open(diag_file);
      MFEM_VERIFY(diag_ofs.is_open(),
                  "Unable to open diagnostics file.");
      diag_ofs.setf(std::ios::scientific, std::ios::floatfield);
      diag_ofs.setf(std::ios::showpos);
      diag_ofs.precision(13);
      diag_ofs << "            time,               cycle,           time_step,"
               << "      kinetic_energy,     internal_energy,           enstrophy,"
               << "                mass,Host Memory Use (GB),"
               << "         total_power,      pressure_power,"
               << "       viscous_power,  solve_total_power,"
               << " solve_pressure_power,  solve_viscous_power,"
               << " solve_conduction_power" << std::endl;
   }
   ConstantCoefficient zero_coeff(0.0);
   const double mass0 = diag_output ? rho0_gf.ComputeL1Error(zero_coeff) : 0.0;

   // Perform time-integration (looping over the time iterations, ti, with a
   // time-step dt). The object oper is of type LagrangianHydroOperator that
   // defines the Mult() method that used by the time integrators.
   ode_solver->Init(*hydro);
   hydro->ResetTimeStepEstimate();
   double t = 0.0;
   double dt = (fixed_dt > 0.0) ? fixed_dt : hydro->GetTimeStepEstimate(S);
   double t_old;
   double next_interp_time = interp_dt;
   bool last_step = false;
   int steps = 0;
   BlockVector S_old(S);
   long mem=0, mmax=0, msum=0;
   int checks = 0;

   if (diag_output)
   {
      double mem_gb = 0.0;
      const double total_power0 = 0.0;
      const double pressure_power0 = 0.0;
      const double viscous_power0 = 0.0;
      if (mem_usage)
      {
         mem = GetMaxRssMB();
         MPI_Reduce(&mem, &mmax, 1, MPI_LONG, MPI_MAX, 0, pmesh->GetComm());
         MPI_Reduce(&mem, &msum, 1, MPI_LONG, MPI_SUM, 0, pmesh->GetComm());
         mem_gb = static_cast<double>(mmax) / 1024.0;
      }
      const double internal_energy0 = hydro->InternalEnergy(e_gf);
      const double kinetic_energy0 = hydro->KineticEnergy(v_gf);
      const double enstrophy0 = ComputeEnstrophy(v_gf);
      if (Mpi::Root())
      {
         const double solve_total_power0 = 0.0;
         const double solve_pressure_power0 = 0.0;
         const double solve_viscous_power0 = 0.0;
         const double solve_conduction_power0 = 0.0;
         diag_ofs << t << ","
                  << 0.0 << ","
                  << dt << ","
                  << kinetic_energy0 << ","
                  << internal_energy0 << ","
                  << enstrophy0 << ","
                  << mass0 << ","
                  << mem_gb << ","
                  << total_power0 << ","
                  << pressure_power0 << ","
                  << viscous_power0 << ","
                  << solve_total_power0 << ","
                  << solve_pressure_power0 << ","
                  << solve_viscous_power0 << ","
                  << solve_conduction_power0 << std::endl;
      }
   }

   //   const double internal_energy = hydro->InternalEnergy(e_gf);
   //   const double kinetic_energy = hydro->KineticEnergy(v_gf);
   //   if (mpi.Root())
   //   {
   //      cout << std::fixed;
   //      cout << "step " << std::setw(5) << 0
   //            << ",\tt = " << std::setw(5) << std::setprecision(4) << t
   //            << ",\tdt = " << std::setw(5) << std::setprecision(6) << dt
   //            << ",\t|IE| = " << std::setprecision(10) << std::scientific
   //            << internal_energy
   //            << ",\t|KE| = " << std::setprecision(10) << std::scientific
   //            << kinetic_energy
   //            << ",\t|E| = " << std::setprecision(10) << std::scientific
   //            << kinetic_energy+internal_energy;
   //      cout << std::fixed;
   //      if (mem_usage)
   //      {
   //         cout << ", mem: " << mmax << "/" << msum << " MB";
   //      }
   //      cout << endl;
   //   }

   std::ofstream csv_ofs;
   if (Mpi::Root())
   {
      csv_ofs.open("laghos_thermo.csv");
      csv_ofs << "                    Time,"
              << "                   Cycle,"
              << "       InternalEnergyAvg,"
              << "            PressureWork,"
              << "             ViscousWork,"
              << "     HeatConductionPower,"
              << " HeatConductionSurfaceFlux,"
              << "                 rho_avg,"
              << "                temp_avg,"
              << "                 rho_rms,"
              << "                temp_rms,"
              << "               div_u_rms,"
              << "                  cs_rms" << std::endl;
      csv_ofs.precision(16);
      csv_ofs << std::scientific;
   }

   for (int ti = 1; !last_step; ti++)
   {
      if (t + dt >= t_final)
      {
         dt = t_final - t;
         last_step = true;
      }
      if (steps == max_tsteps) { last_step = true; }
      S_old = S;
      t_old = t;
      hydro->ResetTimeStepEstimate();

      // S is the vector of dofs, t is the current time, and dt is the time step
      // to advance.
      ode_solver->Step(S, t, dt);
      steps++;

      if (fixed_dt <= 0.0)
      {
         // Adaptive time step control.
         const double dt_est = hydro->GetTimeStepEstimate(S);
         if (dt_est < dt)
         {
            // Repeat (solve again) with a decreased time step - decrease of the
            // time estimate suggests appearance of oscillations.
            dt *= 0.85;
            if (dt < std::numeric_limits<double>::epsilon())
            { MFEM_ABORT("The time step crashed!"); }
            t = t_old;
            S = S_old;
            hydro->ResetQuadratureData();
            if (Mpi::Root()) { cout << "Repeating step " << ti << endl; }
            if (steps < max_tsteps) { last_step = false; }
            ti--; continue;
         }
         else if (dt_est > 1.25 * dt) { dt *= 1.02; }
      }

      // Ensure the sub-vectors x_gf, v_gf, and e_gf know the location of the
      // data in S. This operation simply updates the Memory validity flags of
      // the sub-vectors to match those of S.
      x_gf.SyncAliasMemory(S);
      v_gf.SyncAliasMemory(S);
      e_gf.SyncAliasMemory(S);

      // Make sure that the mesh corresponds to the new solution state. This is
      // needed, because some time integrators use different S-type vectors
      // and the oper object might have redirected the mesh positions to those.
      pmesh->NewNodes(x_gf, false);

      const bool vis_step = last_step || (ti % vis_steps) == 0;
      const bool log_step = vis_step || diag_output;
      double sqrt_norm = 0.0;
      double internal_energy = 0.0;
      double kinetic_energy = 0.0;
      double total_work = 0.0;
      double pressure_work = 0.0;
      double viscous_work = 0.0;
      double conduction_flux = 0.0;
      double enstrophy = 0.0;
      double e_integral = 0.0;
      if (log_step)
      {
         double lnorm = e_gf * e_gf, norm;
         MPI_Allreduce(&lnorm, &norm, 1, MPI_DOUBLE, MPI_SUM, pmesh->GetComm());
         sqrt_norm = sqrt(norm);
         if (mem_usage)
         {
            mem = GetMaxRssMB();
            MPI_Reduce(&mem, &mmax, 1, MPI_LONG, MPI_MAX, 0, pmesh->GetComm());
            MPI_Reduce(&mem, &msum, 1, MPI_LONG, MPI_SUM, 0, pmesh->GetComm());
         }
         internal_energy = hydro->InternalEnergy(e_gf);
         kinetic_energy = hydro->KineticEnergy(v_gf);
         // Compute work contributions over this timestep
         total_work = hydro->ComputeTotalWork(v_gf, dt);
         pressure_work = hydro->ComputePressureWork(v_gf, dt);
         viscous_work = hydro->ComputeViscousWork(v_gf, dt);
         if (use_conduction && pmesh->GetNBE() > 0)
         {
            Diagnostics::ProjectL2toH1(e_gf, e_h1);
            conduction_flux =
               Diagnostics::ComputeConductionSurfaceFlux(e_h1, kappa_e);
         }
         if (diag_output) { enstrophy = ComputeEnstrophy(v_gf); }
         if (track_e_integral)
         {
            const double loc_integral = e_int_lf(e_gf);
            MPI_Allreduce(&loc_integral, &e_integral, 1, MPI_DOUBLE, MPI_SUM,
                          pmesh->GetComm());
         }
      }
      if (Mpi::Root())
      {
         if (vis_step)
         {
            cout << std::fixed;
            cout << "step " << std::setw(5) << ti
                 << ",\tt = " << std::setw(5) << std::setprecision(4) << t
                 << ",\tdt = " << std::setw(5) << std::setprecision(6) << dt
                 << ",\t|e| = " << std::setprecision(10) << std::scientific
                 << sqrt_norm
                 << ",\t|IE| = " << std::setprecision(10) << std::scientific
                 << internal_energy
                 << ",\t|KE| = " << std::setprecision(10) << std::scientific
                 << kinetic_energy
                 << ",\t|E| = " << std::setprecision(10) << std::scientific
                 << kinetic_energy+internal_energy;
            if (track_e_integral)
            {
               cout << ",\tintegral(e) = " << std::setprecision(10)
                    << std::scientific << e_integral;
            }
            cout << std::fixed;
            if (mem_usage)
            {
               cout << ", mem: " << mmax << "/" << msum << " MB";
            }
            cout << endl;
         }
         if (log_step)
         {
            if (hydro->HasSolveEnergyPower())
            {
               const double total_power = hydro->GetSolveEnergyTotalPower();
               const double pressure_power = hydro->GetSolveEnergyPressurePower();
               const double viscous_power = hydro->GetSolveEnergyViscousPower();
               const double conduction_power = hydro->GetSolveEnergyConductionPower();
               const double sum_power = pressure_power + viscous_power + conduction_power;
               cout << "[power] dt=" << std::scientific << std::setprecision(6) << dt
                    << ", total=" << total_power
                    << ", pressure=" << pressure_power
                    << ", viscous=" << viscous_power
                    << ", conduction=" << conduction_power
                    << ", sum=" << sum_power
                    << ", verification=" << (total_power - sum_power)
                    << endl;
            }
         }
      }

      if (log_step)
      {
         double vol, r_avg, t_avg, r_rms, t_rms, d_rms, c_rms;
         hydro->ComputeL2Diagnostics(S, vol, r_avg, t_avg, r_rms, t_rms, d_rms, c_rms);
         if (Mpi::Root())
         {
            const double ie_avg = internal_energy / vol;
            const double p_dil = hydro->GetSolveEnergyPressurePower();
            const double v_dis = hydro->GetSolveEnergyViscousPower();
            const double h_con = hydro->GetSolveEnergyConductionPower();

            csv_ofs << std::setw(24) << t << ", "
                    << std::setw(24) << static_cast<double>(ti) << ", "
                    << std::setw(24) << ie_avg << ", "
                    << std::setw(24) << p_dil << ", "
                    << std::setw(24) << v_dis << ", "
                    << std::setw(24) << h_con << ", "
                    << std::setw(24) << conduction_flux << ", "
                    << std::setw(24) << r_avg << ", "
                    << std::setw(24) << t_avg << ", "
                    << std::setw(24) << r_rms << ", "
                    << std::setw(24) << t_rms << ", "
                    << std::setw(24) << d_rms << ", "
                    << std::setw(24) << c_rms << std::endl;
         }
      }

      if (Mpi::Root())
      {
         if (diag_output)
         {
            const double mem_gb =
               mem_usage ? static_cast<double>(mmax) / 1024.0 : 0.0;
            const double inv_dt = (dt > 0.0) ? 1.0 / dt : 0.0;
            const double total_power = total_work * inv_dt;
            const double pressure_power = pressure_work * inv_dt;
            const double viscous_power = viscous_work * inv_dt;
            const double solve_total_power =
               hydro->GetSolveEnergyTotalPower();
            const double solve_pressure_power =
               hydro->GetSolveEnergyPressurePower();
            const double solve_viscous_power =
               hydro->GetSolveEnergyViscousPower();
            const double solve_conduction_power =
               hydro->GetSolveEnergyConductionPower();
            diag_ofs << t << ","
                     << static_cast<double>(ti) << ","
                     << dt << ","
                     << kinetic_energy << ","
                     << internal_energy << ","
                     << enstrophy << ","
                     << mass0 << ","
                     << mem_gb << ","
                     << total_power << ","
                     << pressure_power << ","
                     << viscous_power << ","
                     << solve_total_power << ","
                     << solve_pressure_power << ","
                     << solve_viscous_power << ","
                     << solve_conduction_power << std::endl;
         }
      }

      if (vis_step)
      {
         // Make sure all ranks have sent their 'v' solution before initiating
         // another set of GLVis connections (one from each rank):
         MPI_Barrier(pmesh->GetComm());

         if (visualization || visit || gfprint) { hydro->ComputeDensity(rho_gf); }
         
         T_gf = e_gf;
         if (NE > 0) { T_gf *= (gamma_func(Vector({0.0})) - 1.0); }

         // Compute Vorticity
         if (visit)
         {
            Diagnostics::ComputeCurl(v_gf, w_gf);
            Diagnostics::ComputeVortexStretching(v_gf, w_gf, w_stretch);
            Diagnostics::ComputeVortexCompression(v_gf, w_gf, w_compress);

            Diagnostics::ProjectL2toH1(rho_gf, rho_h1);
            Diagnostics::ProjectL2toH1(e_gf, e_h1);
            Diagnostics::ComputeBaroclinicTerm(rho_h1, e_h1, w_baroclinic);
            hydro->ComputeWorkFields(v_gf, work_p, work_tau, work_total);
            hydro->ComputeAcceleration(accel_gf, &accel_p, &accel_tau);

            if (visc)
            {
               Diagnostics::ComputeViscousTorqueComponents(rho_h1, accel_tau,
                                                           w_viscous, w_visc_diff, w_visc_baro);
            }

            Diagnostics::ComputeJacobian(const_cast<ParMesh&>(*pmesh),
                                            detJ_gf, logdetJ_gf,
                                            Jfrob_gf, Jinvfrob_gf, cond_gf,
                                            Jcol_gf,
                                            qorder);

            // Verification: Check if accel_total == accel_p + accel_tau
            ParGridFunction accel_diff(accel_gf); // Initialize with Total
            accel_diff.Add(-1.0, accel_p);        // Subtract Pressure
            accel_diff.Add(-1.0, accel_tau);      // Subtract Viscous
            double accel_err = accel_diff.Norml2();
            
            if (Mpi::Root())
            {
               std::cout << "[Accel-Verify] || a_total - (a_p + a_tau) ||_L2 = " 
                         << accel_err << std::endl;
            }

            // if (visc)
            // {
            //    hydro->ComputeViscousAcceleration(S, v_visc_gf);
            //    ComputeCurl(v_visc_gf, w_viscous);
            // }
         }

         if (visualization)
         {
            int Wx = 0, Wy = 0; // window position
            int Ww = 350, Wh = 350; // window size
            int offx = Ww+10; // window offsets
            if (problem != 0 && problem != 4)
            {
               hydrodynamics::VisualizeField(vis_rho, vishost, visport, rho_gf,
                                             "Density", Wx, Wy, Ww, Wh);
            }
            Wx += offx;
            hydrodynamics::VisualizeField(vis_v, vishost, visport,
                                          v_gf, "Velocity", Wx, Wy, Ww, Wh);
            Wx += offx;
            hydrodynamics::VisualizeField(vis_e, vishost, visport, e_gf,
                                          "Specific Internal Energy",
                                          Wx, Wy, Ww,Wh);
            Wx += offx;
         }

         if (visit)
         {
            visit_dc.SetCycle(ti);
            visit_dc.SetTime(t);
            visit_dc.Save();

            if (visc)
            {
               visit_dc_debug.RegisterField("ViscousTorqueTotal", &w_viscous);
               visit_dc_debug.RegisterField("ViscousDiffusionVorticity", &w_visc_diff);
               visit_dc_debug.RegisterField("ViscousBaroclinic", &w_visc_baro);
            }

            visit_dc_debug.SetCycle(ti);
            visit_dc_debug.SetTime(t);
            visit_dc_debug.Save();
         }

         if (gfprint)
         {
            std::ostringstream mesh_name, rho_name, v_name, e_name;
            mesh_name << basename << "_" << ti << "_mesh";
            rho_name  << basename << "_" << ti << "_rho";
            v_name << basename << "_" << ti << "_v";
            e_name << basename << "_" << ti << "_e";

            std::ofstream mesh_ofs(mesh_name.str().c_str());
            mesh_ofs.precision(8);
            pmesh->PrintAsOne(mesh_ofs);
            mesh_ofs.close();

            std::ofstream rho_ofs(rho_name.str().c_str());
            rho_ofs.precision(8);
            rho_gf.SaveAsOne(rho_ofs);
            rho_ofs.close();

            std::ofstream v_ofs(v_name.str().c_str());
            v_ofs.precision(8);
            v_gf.SaveAsOne(v_ofs);
            v_ofs.close();

            std::ofstream e_ofs(e_name.str().c_str());
            e_ofs.precision(8);
            e_gf.SaveAsOne(e_ofs);
            e_ofs.close();
         }
      }

      // Problems checks
      if (check)
      {
         double lnorm = e_gf * e_gf, norm;
         MPI_Allreduce(&lnorm, &norm, 1, MPI_DOUBLE, MPI_SUM, pmesh->GetComm());
         const double e_norm = sqrt(norm);
         MFEM_VERIFY(rs_levels==0 && rp_levels==0, "check: rs, rp");
         MFEM_VERIFY(order_v==2, "check: order_v");
         MFEM_VERIFY(order_e==1, "check: order_e");
         MFEM_VERIFY(ode_solver_type==4, "check: ode_solver_type");
         MFEM_VERIFY(t_final == 0.6, "check: t_final");
         MFEM_VERIFY(cfl==0.5, "check: cfl");
         MFEM_VERIFY(strncmp(mesh_file, "default", 7) == 0, "check: mesh_file");
         MFEM_VERIFY(dim==2 || dim==3, "check: dimension");
         Checks(ti, e_norm, checks);
      }

      if (interp_enabled)
      {
         bool do_remap = false;
         if (interp_cycle_on && (ti % interp_cycle == 0)) { do_remap = true; }
         if (interp_time_on && (t + 1.0e-12) >= next_interp_time)
         {
            do_remap = true;
            while ((t + 1.0e-12) >= next_interp_time)
            {
               next_interp_time += interp_dt;
            }
         }

         if (do_remap)
         {
            if (Mpi::Root())
            {
               cout << "Remap at step " << ti << ", t=" << t << endl;
            }

            ParMesh src_mesh(*pmesh);
            ParMesh tar_mesh(*pmesh);
            if (!tar_mesh.GetNodes())
            {
               tar_mesh.SetCurvature(order_v);
            }
            *tar_mesh.GetNodes() = x0_gf;

            hydro->ComputeDensity(rho_gf);

            std::unique_ptr<ParGridFunction> v_interp(
               InterpolateFieldPeriodic(src_mesh, v_gf, tar_mesh,
                                        0, order_v, 0.0, 0.0, 0.0, false));
            std::unique_ptr<ParGridFunction> e_interp(
               InterpolateFieldPeriodic(src_mesh, e_gf, tar_mesh,
                                        1, order_e, 0.0, 0.0, 0.0, false));
            std::unique_ptr<ParGridFunction> rho_interp(
               InterpolateFieldPeriodic(src_mesh, rho_gf, tar_mesh,
                                        1, order_e, 0.0, 0.0, 0.0, false));

            x_gf = x0_gf;
            x_gf.SyncAliasMemory(S);
            pmesh->NewNodes(x_gf, false);

            v_gf = *v_interp;
            e_gf = *e_interp;
            rho0_gf = *rho_interp;
            v_gf.SyncAliasMemory(S);
            e_gf.SyncAliasMemory(S);

            mat_gf.ProjectCoefficient(mat_coeff);

            hydro = std::make_unique<hydrodynamics::LagrangianHydroOperator>(
               S.Size(),
               H1FESpace, L2FESpace, ess_tdofs,
               rho0_coeff, rho0_gf,
               mat_gf, source, cfl,
               visc, vorticity, viscosity_const,
               use_conduction, prandtl_number,
               freeze_momentum,
               p_assembly,
               cg_tol, cg_max_iter, ftz_tol,
               order_q);
            ode_solver->Init(*hydro);
            hydro->ResetTimeStepEstimate();
            if (fixed_dt <= 0.0)
            {
               dt = hydro->GetTimeStepEstimate(S);
            }
         }
      }
   }
   MFEM_VERIFY(!check || checks == 2, "Check error!");

   switch (ode_solver_type)
   {
      case 2: steps *= 2; break;
      case 3: steps *= 3; break;
      case 4: steps *= 4; break;
      case 6: steps *= 6; break;
      case 7: steps *= 2;
   }

   hydro->PrintTimingData(Mpi::Root(), steps, fom);

   if (mem_usage)
   {
      mem = GetMaxRssMB();
      MPI_Reduce(&mem, &mmax, 1, MPI_LONG, MPI_MAX, 0, pmesh->GetComm());
      MPI_Reduce(&mem, &msum, 1, MPI_LONG, MPI_SUM, 0, pmesh->GetComm());
   }

   const double energy_final = hydro->InternalEnergy(e_gf) +
                               hydro->KineticEnergy(v_gf);
   if (Mpi::Root())
   {
      cout << endl;
      cout << "Energy  diff: " << std::scientific << std::setprecision(2)
           << fabs(energy_init - energy_final) << endl;
      if (mem_usage)
      {
         cout << "Maximum memory resident set size: "
              << mmax << "/" << msum << " MB" << endl;
      }
   }

   // Print the error.
   // For problems 0 and 4 the exact velocity is constant in time.
   if (problem == 0 || problem == 4)
   {
      const double error_max = v_gf.ComputeMaxError(v_coeff),
                   error_l1  = v_gf.ComputeL1Error(v_coeff),
                   error_l2  = v_gf.ComputeL2Error(v_coeff);
      if (Mpi::Root())
      {
         cout << "L_inf  error: " << error_max << endl
              << "L_1    error: " << error_l1 << endl
              << "L_2    error: " << error_l2 << endl;
      }
   }

   if (visualization)
   {
      vis_v.close();
      vis_e.close();
   }

   double vol, r_avg, t_avg, r_rms, t_rms, d_rms, c_rms;
   hydro->ComputeL2Diagnostics(S, vol, r_avg, t_avg, r_rms, t_rms, d_rms, c_rms);

   if (Mpi::Root())
   {
      cout << "\nFINAL L2 DIAGNOSTICS\n";
      cout << "Volume    = " << vol << "\n";
      cout << "rho avg   = " << r_avg << "\n";
      cout << "temp avg  = " << t_avg << "\n";
      cout << "rho RMS   = " << r_rms << "\n";
      cout << "temp RMS  = " << t_rms << "\n";
      cout << "div u RMS = " << d_rms << "\n";
      cout << "cs RMS    = " << c_rms << "\n\n";
   }

   // Free the used memory.
   delete ode_solver;
   delete pmesh;

   MPI_Finalize();

   return 0;
}

double rho0(const Vector &x)
{
   switch (problem)
   {
      case 0: return 1.0;
      case 1: return 1.0;
      case 2: return (x(0) < 0.5) ? 1.0 : 0.1;
      case 3: return (dim == 2) ? (x(0) > 1.0 && x(1) > 1.5) ? 0.125 : 1.0
                        : x(0) > 1.0 && ((x(1) < 1.5 && x(2) < 1.5) ||
                                         (x(1) > 1.5 && x(2) > 1.5)) ? 0.125 : 1.0;
      case 4: return 1.0;
      case 5:
      {
         if (x(0) >= 0.5 && x(1) >= 0.5) { return 0.5313; }
         if (x(0) <  0.5 && x(1) <  0.5) { return 0.8; }
         return 1.0;
      }
      case 6:
      {
         if (x(0) <  0.5 && x(1) >= 0.5) { return 2.0; }
         if (x(0) >= 0.5 && x(1) <  0.5) { return 3.0; }
         return 1.0;
      }
      case 7: return x(1) >= 0.0 ? 2.0 : 1.0;
      case 8: return 1.0;
      default: MFEM_ABORT("Bad number given for problem id!"); return 0.0;
   }
}

double gamma_func(const Vector &x)
{
   switch (problem)
   {
      case 0: return 5.0 / 3.0;
      case 1: return 1.4;
      case 2: return 1.4;
      case 3:
         if (dim == 1) { return (x(0) > 0.5) ? 1.4 : 1.5; }
         else { return (x(0) > 1.0 && x(1) <= 1.5) ? 1.4 : 1.5; }
      case 4: return 5.0 / 3.0;
      case 5: return 1.4;
      case 6: return 1.4;
      case 7: return 5.0 / 3.0;
      case 8: return 5.0 / 3.0;
      default: MFEM_ABORT("Bad number given for problem id!"); return 0.0;
   }
}

static double rad(double x, double y) { return sqrt(x*x + y*y); }

void v0(const Vector &x, Vector &v)
{
   const double atn = dim!=1 ? pow((x(0)*(1.0-x(0))*4*x(1)*(1.0-x(1))*4.0),
                                   0.4) : 0.0;
   switch (problem)
   {
      case 0:
         v(0) =  sin(2.0*M_PI*x(0)) * cos(2.0*M_PI*x(1));
         v(1) = -cos(2.0*M_PI*x(0)) * sin(2.0*M_PI*x(1));
         if (x.Size() == 3)
         {
            v(0) *= cos(2.0*M_PI*x(2));
            v(1) *= cos(2.0*M_PI*x(2));
            v(2) = 0.0;
         }
         break;
      case 1: v = 0.0; break;
      case 2: v = 0.0; break;
      case 3: v = 0.0; break;
      case 4:
      {
         v = 0.0;
         const double r = rad(x(0), x(1));
         if (r < 0.2)
         {
            v(0) =  5.0 * x(1);
            v(1) = -5.0 * x(0);
         }
         else if (r < 0.4)
         {
            v(0) =  2.0 * x(1) / r - 5.0 * x(1);
            v(1) = -2.0 * x(0) / r + 5.0 * x(0);
         }
         else { }
         break;
      }
      case 5:
      {
         v = 0.0;
         if (x(0) >= 0.5 && x(1) >= 0.5) { v(0)=0.0*atn, v(1)=0.0*atn; return;}
         if (x(0) <  0.5 && x(1) >= 0.5) { v(0)=0.7276*atn, v(1)=0.0*atn; return;}
         if (x(0) <  0.5 && x(1) <  0.5) { v(0)=0.0*atn, v(1)=0.0*atn; return;}
         if (x(0) >= 0.5 && x(1) <  0.5) { v(0)=0.0*atn, v(1)=0.7276*atn; return; }
         MFEM_ABORT("Error in problem 5!");
         return;
      }
      case 6:
      {
         v = 0.0;
         if (x(0) >= 0.5 && x(1) >= 0.5) { v(0)=+0.75*atn, v(1)=-0.5*atn; return;}
         if (x(0) <  0.5 && x(1) >= 0.5) { v(0)=+0.75*atn, v(1)=+0.5*atn; return;}
         if (x(0) <  0.5 && x(1) <  0.5) { v(0)=-0.75*atn, v(1)=+0.5*atn; return;}
         if (x(0) >= 0.5 && x(1) <  0.5) { v(0)=-0.75*atn, v(1)=-0.5*atn; return;}
         MFEM_ABORT("Error in problem 6!");
         return;
      }
      case 7:
      {
         v = 0.0;
         v(1) = 0.02 * exp(-2*M_PI*x(1)*x(1)) * cos(2*M_PI*x(0));
         break;
      }
      case 8: v = 0.0; break;
      default: MFEM_ABORT("Bad number given for problem id!");
   }
}

double e0(const Vector &x)
{
   switch (problem)
   {
      case 0:
      {
         const double rho = rho0(x);
         const double gamma = gamma_func(x);
         const double denom = (gamma - 1.0) * rho;
         const double L = 1.0 / (2.0 * M_PI);
         const double k = 2.0 / L;
         const double p0 = p0_background;
         const double rho0u0u0 = rho * mach_u0 * mach_u0;
         double val;
         if (x.Size() == 2)
         {
            val = p0 + (rho0u0u0 / 4.0) *
                        (cos(k*x(0)) + cos(k*x(1)));
         }
         else
         {
            val = p0 + (rho0u0u0 / 16.0) *
                        (cos(k*x(0)) + cos(k*x(1))) *
                        (cos(k*x(2)) + 2.0);
         }
         return val / denom;
      }
      case 1: return 0.0; // This case in initialized in main().
      case 2: return (x(0) < 0.5) ? 1.0 / rho0(x) / (gamma_func(x) - 1.0)
                        : 0.1 / rho0(x) / (gamma_func(x) - 1.0);
      case 3: return (x(0) > 1.0) ? 0.1 / rho0(x) / (gamma_func(x) - 1.0)
                        : 1.0 / rho0(x) / (gamma_func(x) - 1.0);
      case 4:
      {
         const double r = rad(x(0), x(1)), rsq = x(0) * x(0) + x(1) * x(1);
         const double gamma = 5.0 / 3.0;
         if (r < 0.2)
         {
            return (5.0 + 25.0 / 2.0 * rsq) / (gamma - 1.0);
         }
         else if (r < 0.4)
         {
            const double t1 = 9.0 - 4.0 * log(0.2) + 25.0 / 2.0 * rsq;
            const double t2 = 20.0 * r - 4.0 * log(r);
            return (t1 - t2) / (gamma - 1.0);
         }
         else { return (3.0 + 4.0 * log(2.0)) / (gamma - 1.0); }
      }
      case 5:
      {
         const double irg = 1.0 / rho0(x) / (gamma_func(x) - 1.0);
         if (x(0) >= 0.5 && x(1) >= 0.5) { return 0.4 * irg; }
         if (x(0) <  0.5 && x(1) >= 0.5) { return 1.0 * irg; }
         if (x(0) <  0.5 && x(1) <  0.5) { return 1.0 * irg; }
         if (x(0) >= 0.5 && x(1) <  0.5) { return 1.0 * irg; }
         MFEM_ABORT("Error in problem 5!");
         return 0.0;
      }
      case 6:
      {
         const double irg = 1.0 / rho0(x) / (gamma_func(x) - 1.0);
         if (x(0) >= 0.5 && x(1) >= 0.5) { return 1.0 * irg; }
         if (x(0) <  0.5 && x(1) >= 0.5) { return 1.0 * irg; }
         if (x(0) <  0.5 && x(1) <  0.5) { return 1.0 * irg; }
         if (x(0) >= 0.5 && x(1) <  0.5) { return 1.0 * irg; }
         MFEM_ABORT("Error in problem 6!");
         return 0.0;
      }
      case 7:
      {
         const double rho = rho0(x), gamma = gamma_func(x);
         return (6.0 - rho * x(1)) / (gamma - 1.0) / rho;
      }
      case 8:
      {
         Vector center(x.Size());
         double min_box_span = std::numeric_limits<double>::max();
         for (int i = 0; i < x.Size(); i++)
         {
            center(i) = 0.5 * (bb_min(i) + bb_max(i));
            min_box_span = std::min(min_box_span, bb_max(i) - bb_min(i));
         }
         const double radius = 0.2 * min_box_span;
         const double radius2 = radius * radius;
         double r2 = 0.0;
         for (int i = 0; i < x.Size(); i++)
         {
            const double dx = x(i) - center(i);
            r2 += dx * dx;
         }
         double T = (r2 <= radius2) ? 2.0 : 0.0;
         return T;
      }
      default: MFEM_ABORT("Bad number given for problem id!"); return 0.0;
   }
}

static void display_banner(std::ostream &os)
{
   os << endl
      << "       __                __                 " << endl
      << "      / /   ____  ____  / /_  ____  _____   " << endl
      << "     / /   / __ `/ __ `/ __ \\/ __ \\/ ___/ " << endl
      << "    / /___/ /_/ / /_/ / / / / /_/ (__  )    " << endl
      << "   /_____/\\__,_/\\__, /_/ /_/\\____/____/  " << endl
      << "               /____/                       " << endl << endl;
}

static void SaveParallelMesh(const ParMesh &pmesh, const char *mesh_dir,
                             const char *fname_base)
{
   if (!mesh_dir || mesh_dir[0] == '\0') { return; }

   if (Mpi::Root())
   {
      std::string cmd = std::string("mkdir -p ") + mesh_dir;
      int ret = system(cmd.c_str());
      MFEM_VERIFY(ret == 0, "Failed to create directory: " + std::string(mesh_dir));
   }

   MPI_Barrier(pmesh.GetComm());

   std::ostringstream mesh_name;
   mesh_name << mesh_dir << "/" << fname_base << "."
             << std::setfill('0') << std::setw(6) << Mpi::WorldRank();

   std::ofstream mesh_ofs(mesh_name.str().c_str());
   MFEM_VERIFY(mesh_ofs.good(),
               "Failed to open mesh file for writing: " + mesh_name.str());
   mesh_ofs.precision(17);
   pmesh.ParPrint(mesh_ofs);
   mesh_ofs.close();

   MPI_Barrier(pmesh.GetComm());
}

static long GetMaxRssMB()
{
   struct rusage usage;
   if (getrusage(RUSAGE_SELF, &usage)) { return -1; }
#ifndef __APPLE__
   const long unit = 1024; // kilo
#else
   const long unit = 1024*1024; // mega
#endif
   return usage.ru_maxrss/unit; // mega bytes
}

static void Checks(const int ti, const double nrm, int &chk)
{
   const double eps = 1.e-13;
   //printf("\033[33m%.15e\033[m\n",nrm);

   auto check = [&](int p, int i, const double res)
   {
      auto rerr = [](const double a, const double v, const double eps)
      {
         MFEM_VERIFY(fabs(a) > eps && fabs(v) > eps, "One value is near zero!");
         const double err_a = fabs((a-v)/a);
         const double err_v = fabs((a-v)/v);
         return fmax(err_a, err_v) < eps;
      };
      if (problem == p && ti == i)
      { chk++; MFEM_VERIFY(rerr(nrm, res, eps), "P"<<problem<<", #"<<i); }
   };

   const double it_norms[2][8][2][2] = // dim, problem, {it,norm}
   {
      {
         {{5, 6.546538624534384e+00}, { 27, 7.588576357792927e+00}},
         {{5, 3.508254945225794e+00}, { 15, 2.756444596823211e+00}},
         {{5, 1.020745795651244e+01}, { 59, 1.721590205901898e+01}},
         {{5, 8.000000000000000e+00}, { 16, 8.000000000000000e+00}},
         {{5, 3.446324942352448e+01}, { 18, 3.446844033767240e+01}},
         {{5, 1.030899557252528e+01}, { 36, 1.057362418574309e+01}},
         {{5, 8.039707010835693e+00}, { 36, 8.316970976817373e+00}},
         {{5, 1.514929259650760e+01}, { 25, 1.514931278155159e+01}},
      },
      {
         {{5, 1.198510951452527e+03}, {188, 1.199384410059154e+03}},
         {{5, 1.339163718592566e+01}, { 28, 7.521073677397994e+00}},
         {{5, 2.041491591302486e+01}, { 59, 3.443180411803796e+01}},
         {{5, 1.600000000000000e+01}, { 16, 1.600000000000000e+01}},
         {{5, 6.892649884704898e+01}, { 18, 6.893688067534482e+01}},
         {{5, 2.061984481890964e+01}, { 36, 2.114519664792607e+01}},
         {{5, 1.607988713996459e+01}, { 36, 1.662736010353023e+01}},
         {{5, 3.029858112572883e+01}, { 24, 3.029858832743707e+01}}
      }
   };

   for (int p=0; p<8; p++)
   {
      for (int i=0; i<2; i++)
      {
         const int it = it_norms[dim-2][p][i][0];
         const double norm = it_norms[dim-2][p][i][1];
         check(p, it, norm);
      }
   }
}

static double ComputeEnstrophy(const ParGridFunction &v_gf)
{
   ParFiniteElementSpace *pfes = v_gf.ParFESpace();
   if (!pfes) { return 0.0; }
   ParMesh *pmesh = pfes->GetParMesh();
   if (!pmesh || pmesh->Dimension() < 2) { return 0.0; }

   const int curl_dim = v_gf.CurlDim();
   Vector curl(curl_dim);
   double local = 0.0;

   for (int e = 0; e < pfes->GetNE(); ++e)
   {
      ElementTransformation *tr = pfes->GetElementTransformation(e);
      const FiniteElement *el = pfes->GetFE(e);
      int ir_order = 2 * el->GetOrder();
      if (ir_order < 2) { ir_order = 2; }
      const IntegrationRule &ir = IntRules.Get(tr->GetGeometryType(), ir_order);
      for (int i = 0; i < ir.GetNPoints(); ++i)
      {
         const IntegrationPoint &ip = ir.IntPoint(i);
         tr->SetIntPoint(&ip);
         v_gf.GetCurl(*tr, curl);
         double curl_sq = 0.0;
         for (int d = 0; d < curl_dim; ++d) { curl_sq += curl(d) * curl(d); }
         local += curl_sq * ip.weight * tr->Weight();
      }
   }

   double global = 0.0;
   MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, pmesh->GetComm());
   return 0.5 * global;
}
