/*

                            Copyright (c) 2014.
               Lawrence Livermore National Security, LLC.
         Produced at the Lawrence Livermore National Laboratory
                             LLNL-CODE-656392.
                           All rights reserved.

This file is part of CoEVP, Version 1.0. Please also read this link -- http://www.opensource.org/licenses/index.php

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

   * Redistributions of source code must retain the above copyright
     notice, this list of conditions and the disclaimer below.

   * Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the disclaimer (as noted below)
     in the documentation and/or other materials provided with the
     distribution.

   * Neither the name of the LLNS/LLNL nor the names of its contributors
     may be used to endorse or promote products derived from this software
     without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY, LLC,
THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Additional BSD Notice

1. This notice is required to be provided under our contract with the U.S.
   Department of Energy (DOE). This work was produced at Lawrence Livermore
   National Laboratory under Contract No. DE-AC52-07NA27344 with the DOE.

2. Neither the United States Government nor Lawrence Livermore National
   Security, LLC nor any of their employees, makes any warranty, express
   or implied, or assumes any liability or responsibility for the accuracy,
   completeness, or usefulness of any information, apparatus, product, or
   process disclosed, or represents that its use would not infringe
   privately-owned rights.

3. Also, reference herein to any specific commercial products, process, or
   services by trade name, trademark, manufacturer or otherwise does not
   necessarily constitute or imply its endorsement, recommendation, or
   favoring by the United States Government or Lawrence Livermore National
   Security, LLC. The views and opinions of authors expressed herein do not
   necessarily state or reflect those of the United States Government or
   Lawrence Livermore National Security, LLC, and shall not be used for
   advertising or product endorsement purposes.

*/
#if defined(MPI_TASK_POOL)

#include <mpi.h>
#include "GammaLawGas.h"
#include "MieGruneisen.h"
#include "Taylor.h"
#include "ElastoViscoPlasticity.h"
#include "ApproxNearestNeighborsMTree.h"
#include "ApproxNearestNeighborsFLANN.h"



void setVelocityGradient(double      time,
                         Tensor2Gen& L)
{
   L = Tensor2Gen(0);

   L(1,1) = L(2,2) = -0.5;
   L(3,3) = 1.;

   L(1,3) = L(3,1) = 1.;
}

int 
main( int   argc,
      char *argv[] )
{
   // Construct the fine-scale plasticity model
   double m = 1./20.;
   double g = 2.e-3;
   double D_0 = 1.e-2;
   Taylor plasticity_model(D_0, m, g);

   // Construct the equation of state
   EOS* eos_model;
   {
      /* From Table 1 in P. J. Maudlin et al., "On the modeling of the
         Taylor cylinder impact test for orthotropic textured materials:
         experiments and simulations", Inter. J. Plasticity 15 (1999),
         pp. 139-166.
      */
      double k1 = 1.968;  // Mbar
      double k2 = 2.598;  // Mbar
      double k3 = 2.566;  // Mbar
      double Gamma = 1.60;  // dimensionless
      eos_model = (EOS*)(new MieGruneisen(k1, k2, k3, Gamma));
   }

   // Construct approximate nearest neighbor search object
   int point_dimension = plasticity_model.pointDimension();
   ApproxNearestNeighbors* ann;

#ifdef FLANN
   int flann_n_trees = 1;
   int flann_n_checks = 20;
   ann = (ApproxNearestNeighbors*)(new ApproxNearestNeighborsFLANN(point_dimension, flann_n_trees, flann_n_checks));
#else
   std::string mtreeDirectoryName = ".";
   ann = (ApproxNearestNeighbors*)(new ApproxNearestNeighborsMTree(point_dimension,
                                                                   "kriging_model_database",
                                                                   mtreeDirectoryName,
                                                                   &(std::cout),
                                                                   false));
#endif

   // Construct the constitutive model
   bool use_adaptive_sampling = false;
   ModelDatabase * modelDB = nullptr;
   Tensor2Gen L_init;
   setVelocityGradient(0., L_init);

   double K = 1.94; // Bulk modulus of Tantallum (Mbar)
   double G = 6.9e-1;  // Shear modulus of Tantallum (Mbar)

   ConstitutiveGlobal cm_global;
   size_t state_size;
   ElastoViscoPlasticity constitutive_model(cm_global, ann, modelDB, L_init, K, G, eos_model, &plasticity_model, use_adaptive_sampling, state_size);

   // Allocate an opaque blob to hold the constitutive model state
   void* state = operator new(state_size);
   constitutive_model.getState(state);

   // Set up the time integration
   double end_time = 2.e-3;
   int num_steps = 2;
   double delta_t = end_time / num_steps;
   double time = 0.;
   int rank;
   int size;
   MPI_Init(&argc, &argv);
   MPI_Comm mpi_comm_taskpool;
   MPI_Comm_dup(MPI_COMM_WORLD, &mpi_comm_taskpool);

   MPI_Comm_rank (mpi_comm_taskpool, &rank);
   MPI_Comm_size (mpi_comm_taskpool, &size);

  // let's build an enormous intracommunicator (hmmm this is ugly and unecessary but I can't thnk of a clean way to do it right now)

  MPI_Comm mpi_intercomm_parent;
  MPI_Comm_get_parent(&mpi_intercomm_parent);

  // let's broadcast the number of task handlers why not, this is using an intercommunicator so behaves a little difference
  
  int numTaskHandlers;
 
  MPI_Bcast(&numTaskHandlers, 1, MPI_INT, 0, mpi_intercomm_parent);
  
  int myHandler = (int) (((float)rank / (float)size) * (float)numTaskHandlers);
 
  printf("Task %d sees that there are %d task handlers. It is affinitised to Task Handler %d\n", rank, numTaskHandlers, myHandler);

 int step=1;
   
   // this is our blocking while task loop that continues to crunch models until the program dies
	int lulesh_worker_id;

	while(1)
	{
      // Advance the hydro, obtaining new values for the following:
   //   Tensor2Gen L_new;
   //   setVelocityGradient(time, L_new);

      // Advance the constitutive model to the new time
   //   ConstitutiveData cm_data = constitutive_model.advance(delta_t, L_new, 1., state);

      //      cout << "Number of Newton iterations = " << cm_data.num_Newton_iters << endl;

      // Print some interpolation statistics if adaptive sampling is being used
   //   constitutive_model.printNewInterpStats();

      // Get the new Cauchy stress and update hydro
  //    const Tensor2Sym& sigma_prime = cm_data.sigma_prime;

  //    time += delta_t;

   //   cout << "Step " << step << " completed, simulation time is " << time << endl;



     // I can't do any work until the handler knows I'm free, so I'll do a blocking send
     // all we are doing with this framework is pairing up tasks with lulesh ranks so we just have to give our rank to the handler

 		MPI_Send(&rank, 1, MPI_INT, myHandler, 2, mpi_intercomm_parent);
     	// If we sent succesfully, then we are ready to discover some work
		MPI_Recv(&lulesh_worker_id, 1, MPI_INT, myHandler, 3, mpi_intercomm_parent, MPI_STATUS_IGNORE);

		printf("Task %d was paired up with Lulesh domain %d\n", rank, lulesh_worker_id);

		MPI_Send(&rank, 1, MPI_INT, lulesh_worker_id, 4, mpi_intercomm_parent);

		// this is where we receive work from the lulesh domain and ultimately send it back

		printf("Task %d is done working with domain %d\n", rank, lulesh_worker_id);
      

   }

   MPI_Finalize();
}

#endif