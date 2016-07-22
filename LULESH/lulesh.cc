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

#include <memory>
#include <vector>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdexcept>
#include <sstream>
#include <climits>

#if defined(COEVP_MPI)
#include <mpi.h>
#endif

#if defined(LOGGER)      // CoEVP Makefile enforces assert LOGGER=REDIS=yes
#include "LoggerDB.h"    // Includes Logger base class too
#include "Locator.h"
#endif

#if defined(MSGPACK)
#include "msgpack.hpp"
#endif

#if defined(PROTOBUF)
#include "shims.h"
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef SILO
#include "siloDump.h"
#endif

int showMeMonoQ = 0 ;

#define PRINT_PERFORMANCE_DIAGNOSTICS
#define LULESH_SHOW_PROGRESS
#undef WRITE_FSM_EVAL_COUNT
#undef WRITE_CHECKPOINT

// Domain
#include "lulesh.h"

// EOS options
#include "BulkPressure.h"
#include "MieGruneisen.h"

// Constitutive model options
#include "IdealGas.h"
#include "ElastoViscoPlasticity.h"

// Approximate nearest neighbor search options
#include "ApproxNearestNeighborsFLANN.h"
#include "ApproxNearestNeighborsMTree.h"

// Database options
#include "ModelDatabase.h"
#include "ModelDB_HashMap.h"
#include "ModelDB_SingletonDB.h"
#include "SingletonDB.h"

// Fine scale model options
#include "Taylor.h"        // the fine-scale plasticity model
#include "vpsc.h"

enum { VolumeError = -1, QStopError = -2 } ;


/* Stuff needed for boundary conditions */
/* 2 BCs on each of 6 hexahedral faces (12 bits) */
#define XI_M        0x0007
#define XI_M_SYMM   0x0001
#define XI_M_FREE   0x0002
#define XI_M_COMM   0x0004

#define XI_P        0x0038
#define XI_P_SYMM   0x0008
#define XI_P_FREE   0x0010
#define XI_P_COMM   0x0020

#define ETA_M       0x00C0
#define ETA_M_SYMM  0x0040
#define ETA_M_FREE  0x0080

#define ETA_P       0x0300
#define ETA_P_SYMM  0x0100
#define ETA_P_FREE  0x0200

#define ZETA_M      0x0C00
#define ZETA_M_SYMM 0x0400
#define ZETA_M_FREE 0x0800

#define ZETA_P      0x3000
#define ZETA_P_SYMM 0x1000
#define ZETA_P_FREE 0x2000

#if defined(COEVP_MPI)||defined(__CHARMC__)

/* Assume 128 byte coherence */
/* Assume Real_t is an "integral power of 2" bytes wide */
#define CACHE_COHERENCE_PAD_REAL (128 / sizeof(Real_t))

#define CACHE_ALIGN_REAL(n) \
   (((n) + (CACHE_COHERENCE_PAD_REAL - 1)) & ~(CACHE_COHERENCE_PAD_REAL-1))

/******************************************/

/* Comm Routines */

#define MAX_FIELDS_PER_MPI_COMM 6

#define MSG_COMM_SBN      1024
#define MSG_SYNC_POS_VEL  2048
#define MSG_MONOQ         3072

/*
 *    define one of these three symbols:
 *
 *    SEDOV_SYNC_POS_VEL_NONE
 *    SEDOV_SYNC_POS_VEL_EARLY
 *    SEDOV_SYNC_POS_VEL_LATE
 */

//#define SEDOV_SYNC_POS_VEL_EARLY 1
#endif

#if defined(COEVP_MPI)
/* doRecv flag only works with regular block structure */
void Lulesh::CommRecv(Domain *domain, int msgType, Index_t xferFields, Index_t size,
      bool recvMin = true)
{

   if (domain->numSlices() == 1) return ;

   /* post recieve buffers for all incoming messages */
   int myRank = domain->sliceLoc() ;
   Index_t maxPlaneComm = MAX_FIELDS_PER_MPI_COMM * domain->maxPlaneSize() ;
   Index_t pmsg = 0 ; /* plane comm msg */
   MPI_Datatype baseType = ((sizeof(Real_t) == 4) ? MPI_FLOAT : MPI_DOUBLE) ;
   bool planeMin, planeMax ;

   /* assume communication to 2 neighbors by default */
   planeMin = planeMax = true ;

   if (domain->sliceLoc() == 0) {
      planeMin = false ;
   }
   if (domain->sliceLoc() == (domain->numSlices()-1)) {
      planeMax = false ;
   }

   for (Index_t i=0; i<2; ++i) {
      domain->recvRequest[i] = MPI_REQUEST_NULL ;
   }

   /* post receives */

   /* receive data from neighboring domain faces */
   if (planeMin && recvMin) {
      /* contiguous memory */
      int fromRank = myRank - 1 ;
      int recvCount = size * xferFields ;
      MPI_Irecv(&domain->commDataRecv[pmsg * maxPlaneComm],
            recvCount, baseType, fromRank, msgType,
            MPI_COMM_WORLD, &domain->recvRequest[pmsg]) ;
      ++pmsg ;
   }
   if (planeMax) {
      /* contiguous memory */
      int fromRank = myRank + 1 ;
      int recvCount = size * xferFields ;
      MPI_Irecv(&domain->commDataRecv[pmsg * maxPlaneComm],
            recvCount, baseType, fromRank, msgType,
            MPI_COMM_WORLD, &domain->recvRequest[pmsg]) ;
      ++pmsg ;
   }
}

void Lulesh::CommSend(Domain *domain, int msgType,
      Index_t xferFields, Real_t **fieldData,
      Index_t *iset,  Index_t size, Index_t offset,
      bool sendMax = true)
{

   if (domain->numSlices() == 1) return ;

   /* post recieve buffers for all incoming messages */
   int myRank = domain->sliceLoc() ;
   Index_t maxPlaneComm = MAX_FIELDS_PER_MPI_COMM * domain->maxPlaneSize() ;
   Index_t pmsg = 0 ; /* plane comm msg */
   MPI_Datatype baseType = ((sizeof(Real_t) == 4) ? MPI_FLOAT : MPI_DOUBLE) ;
   MPI_Status status[2] ;
   Real_t *destAddr ;
   bool planeMin, planeMax ;
   /* assume communication to 2 neighbors by default */
   planeMin = planeMax = true ;
   if (domain->sliceLoc() == 0) {
      planeMin = false ;
   }
   if (domain->sliceLoc() == (domain->numSlices()-1)) {
      planeMax = false ;
   }

   for (Index_t i=0; i<2; ++i) {
      domain->sendRequest[i] = MPI_REQUEST_NULL ;
   }

   /* post sends */

   /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */

   if (planeMin) {
      destAddr = &domain->commDataSend[pmsg * maxPlaneComm] ;
      if (showMeMonoQ) {
         printf("%d, %d, %d -> %d: ", domain->cycle(), offset, domain->sliceLoc(), domain->sliceLoc() - 1) ;
      }
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Real_t *srcAddr = fieldData[fi] ;
         for (Index_t ii=0; ii<size; ++ii) {
            destAddr[ii] = srcAddr[iset[ii]] ;
            if (showMeMonoQ) {
               printf("%e[%d] ", srcAddr[iset[ii]], iset[ii]) ;
            }
         }
         destAddr += size ;
      }
      destAddr -= xferFields*size ;
      if (showMeMonoQ) {
         printf("\n") ;
      }

      MPI_Isend(destAddr, xferFields*size,
            baseType, myRank - 1, msgType,
            MPI_COMM_WORLD, &domain->sendRequest[pmsg]) ;
      ++pmsg ;
   }

   if (planeMax && sendMax) {
      destAddr = &domain->commDataSend[pmsg * maxPlaneComm] ;
      if (showMeMonoQ) {
         printf("%d, %d, %d -> %d: ", domain->cycle(), offset, domain->sliceLoc(), domain->sliceLoc() + 1) ;
      }
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Real_t *srcAddr = &fieldData[fi][offset] ;
         for (Index_t ii=0; ii<size; ++ii) {
            destAddr[ii] = srcAddr[iset[ii]] ;
            if (showMeMonoQ) {
               printf("%e[%d] ", srcAddr[iset[ii]], iset[ii]+offset) ;
            }
         }
         destAddr += size ;
      }
      destAddr -= xferFields*size ;
      if (showMeMonoQ) {
         printf("\n") ;
      }

      MPI_Isend(destAddr, xferFields*size,
            baseType, myRank + 1, msgType,
            MPI_COMM_WORLD, &domain->sendRequest[pmsg]) ;
      ++pmsg ;
   }

   MPI_Waitall(2, domain->sendRequest, status) ;
}


void Lulesh::CommSBN(Domain *domain, int xferFields, Real_t **fieldData,
      Index_t *iset, Index_t size, Index_t offset) {

   if (domain->numSlices() == 1) return ;

   /* summation order should be from smallest value to largest */
   /* or we could try out kahan summation! */

   int myRank = domain->sliceLoc() ;
   Index_t maxPlaneComm = MAX_FIELDS_PER_MPI_COMM * domain->maxPlaneSize() ;
   Index_t pmsg = 0 ; /* plane comm msg */
   MPI_Status status ;
   Real_t *srcAddr ;
   Index_t planeMin, planeMax ;
   /* assume communication to 2 neighbors by default */
   planeMin = planeMax = 1 ;
   if (domain->sliceLoc() == 0) {
      planeMin = 0 ;
   }
   if (domain->sliceLoc() == (domain->numSlices()-1)) {
      planeMax = 0 ;
   }

   /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */

   if (planeMin) {
      srcAddr = &domain->commDataRecv[pmsg * maxPlaneComm] ;
      MPI_Wait(&domain->recvRequest[pmsg], &status) ;
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Real_t *destAddr = fieldData[fi] ;
         for (Index_t i=0; i<size; ++i) {
            destAddr[iset[i]] += srcAddr[i] ;

         }
         srcAddr += size ;
      }
      ++pmsg ;
   }
   if (planeMax) {
      srcAddr = &domain->commDataRecv[pmsg * maxPlaneComm] ;
      MPI_Wait(&domain->recvRequest[pmsg], &status) ;
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Real_t *destAddr = &fieldData[fi][offset] ;
         for (Index_t i=0; i<size; ++i) {
            destAddr[iset[i]] += srcAddr[i] ;
#ifdef COMM_TEST
            if (domain->sliceLoc() == 0)
                printf("receiveDataNodes: %d %d %d %d from P1  %d %d %d = %.10e %.10e\n", domain->sliceLoc(), xferFields, fi, offset, size, i, iset[i], srcAddr[i], destAddr[iset[i]]);
#endif
         }
         srcAddr += size ;
      }
      ++pmsg ;
   }
}


void Lulesh::CommSyncPosVel(Domain *domain,
      Index_t *iset, Index_t size, Index_t offset)
{

   if (domain->numSlices() == 1) return ;

   int myRank = domain->sliceLoc() ;
   Index_t xferFields = 6 ; /* x, y, z, xd, yd, zd */
   Real_t *fieldData[6] ;
   Index_t maxPlaneComm = MAX_FIELDS_PER_MPI_COMM * domain->maxPlaneSize() ;
   Index_t pmsg = 0 ; /* plane comm msg */
   MPI_Status status ;
   Real_t *srcAddr ;
   bool planeMin, planeMax ;
   /* assume communication to 2 neighbors by default */
   planeMin = planeMax = true ;
   if (domain->sliceLoc() == 0) {
      planeMin = false ;
   }
   if (domain->sliceLoc() == (domain->numSlices()-1)) {
      planeMax = false ;
   }

   fieldData[0] = &domain->x(0) ;
   fieldData[1] = &domain->y(0) ;
   fieldData[2] = &domain->z(0) ;
   fieldData[3] = &domain->xd(0) ;
   fieldData[4] = &domain->yd(0) ;
   fieldData[5] = &domain->zd(0) ;

#if 0
   if (planeMin) {
      /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */

      srcAddr = &domain->commDataRecv[pmsg * maxPlaneComm] ;
      MPI_Wait(&domain->recvRequest[pmsg], &status) ;
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Real_t *destAddr = fieldData[fi] ;
         for (Index_t i=0; i<size; ++i) {
            destAddr[iset[i]] = srcAddr[i] ;
         }
         srcAddr += size ;
      }
      ++pmsg ;
   }
#endif

   if (planeMax) {
      srcAddr = &domain->commDataRecv[pmsg * maxPlaneComm] ;
      MPI_Wait(&domain->recvRequest[pmsg], &status) ;
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Real_t *destAddr = &fieldData[fi][offset] ;
         for (Index_t i=0; i<size; ++i) {
            destAddr[iset[i]] = srcAddr[i] ;
#ifdef COMM_TEST
            if (domain->sliceLoc() == 0)
                printf("receiveDataNodes: %d %d %d %d from P1  %d %d %d = %.10e %.10e\n", domain->sliceLoc(), xferFields, fi, offset, size, i, iset[i], srcAddr[i], destAddr[iset[i]]);
#endif
         }
         srcAddr += size ;
      }
      ++pmsg ;
   }
}

#endif


#if defined(COEVP_MPI)

void Lulesh::CommMonoQ(Domain *domain, Index_t *iset, Index_t size, Index_t offset)
{
   if (domain->numSlices() == 1) return ;

   int myRank = domain->sliceLoc() ;
   // Index_t xferFields = 3 ; /* delv_xi, delv_eta, delv_zeta */
   // Real_t *fieldData[3] ;
   Index_t xferFields = 1 ; /* delv_xi, delv_eta, delv_zeta */
   Real_t *fieldData[1] ;
   Index_t maxPlaneComm = MAX_FIELDS_PER_MPI_COMM * domain->maxPlaneSize() ;
   Index_t pmsg = 0 ; /* plane comm msg */
   MPI_Status status ;
   Real_t *srcAddr ;
   bool planeMin, planeMax ;
   /* assume communication to 6 neighbors by default */
   planeMin = planeMax = true ;
   if (domain->sliceLoc() == 0) {
      planeMin = false ;
   }
   if (domain->sliceLoc() == (domain->numSlices()-1)) {
      planeMax = false ;
   }

   /* point into ghost data area */
   fieldData[0] = &domain->delv_xi(domain->numElem()) ;
   // fieldData[1] = &domain->delv_eta(domain->numElem()) ;
   // fieldData[2] = &domain->delv_zeta(domain->numElem()) ;

   /* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      NOT CLEAR IF WE SHOULD UNPACK LINEARLY
      OR WITH RESPECT TO ISET
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

   if (planeMin) {
      /* contiguous memory */
      srcAddr = &domain->commDataRecv[pmsg * maxPlaneComm] ;
      MPI_Wait(&domain->recvRequest[pmsg], &status) ;
      if (showMeMonoQ) {
         printf("%d, %d, %d <- %d: ", domain->cycle(), offset, domain->sliceLoc(), domain->sliceLoc() - 1) ;
      }
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Real_t *destAddr = fieldData[fi] ;
         for (Index_t i=0; i<size; ++i) {
            destAddr[i] = srcAddr[i] ;

            if (showMeMonoQ) {
               printf("%e[%d] ", srcAddr[i], iset[i]) ;
            }
         }
         srcAddr += size ;
         fieldData[fi] += size ; /* prepare each field for next plane */
      }
      if (showMeMonoQ) {
         printf("\n") ;
      }
      ++pmsg ;
   }
   if (planeMax) {
      /* contiguous memory */
      srcAddr = &domain->commDataRecv[pmsg * maxPlaneComm] ;
      MPI_Wait(&domain->recvRequest[pmsg], &status) ;
      if (showMeMonoQ) {
      //   printf("%d, %d, %d <- %d: ", domain->cycle(), offset, domain->sliceLoc(), domain->sliceLoc() + 1) ;
      }
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Real_t *destAddr = fieldData[fi] ;
         for (Index_t i=0; i<size; ++i) {
            destAddr[i] = srcAddr[i] ;

#ifdef COMM_TEST
            if (domain->sliceLoc() == 0)
                printf("receiveDataElems: %d %d %d %d from P1  %d %d %d = %.10e\n", domain->sliceLoc(), xferFields, fi, offset, size, i, i, srcAddr[i]);
#endif
            if (showMeMonoQ) {
               printf("%e[%d] ", srcAddr[i], iset[i]+offset) ;
            }
         }
         srcAddr += size ;
         fieldData[fi] += size ;
      }
      if (showMeMonoQ) {
         printf("\n") ;
      }
      ++pmsg ;
   }
}


#endif

// Factor to be multiply the time step by to compensate
// for fast time scales in the fine-scale model
Real_t finescale_dt_modifier = Real_t(1.);

void Lulesh::TimeIncrement()
{
   Real_t targetdt = domain.stoptime() - domain.time() ;

   if ((domain.dtfixed() <= Real_t(0.0)) && (domain.cycle() != Int_t(0))) {
      Real_t ratio ;
      Real_t olddt = domain.deltatime() ;

      /* This will require a reduction in parallel */
      Real_t gnewdt = Real_t(1.0e+20) ;
      Real_t newdt ;
      if (domain.dtcourant() < gnewdt) {
         gnewdt = domain.dtcourant() / Real_t(2.0) ;
      }
      if (domain.dthydro() < gnewdt) {
         gnewdt = domain.dthydro() * Real_t(2.0) / Real_t(3.0) ;
      }

      gnewdt *= finescale_dt_modifier;

#if defined(COEVP_MPI)
      MPI_Allreduce(&gnewdt, &newdt, 1,
            ((sizeof(Real_t) == 4) ? MPI_FLOAT : MPI_DOUBLE),
            MPI_MIN, MPI_COMM_WORLD) ;
#else
      newdt = gnewdt ;
#endif
	this->OutputTiming();

      ratio = newdt / olddt ;
      if (ratio >= Real_t(1.0)) {
         if (ratio < domain.deltatimemultlb()) {
            newdt = olddt ;
         }
         else if (ratio > domain.deltatimemultub()) {
            newdt = olddt*domain.deltatimemultub() ;
         }
      }

      if (newdt > domain.dtmax()) {
         newdt = domain.dtmax() ;
      }
      domain.deltatime() = newdt ;
   }

   /* TRY TO PREVENT VERY SMALL SCALING ON THE NEXT CYCLE */
   if ((targetdt > domain.deltatime()) &&
         (targetdt < (Real_t(4.0) * domain.deltatime() / Real_t(3.0))) ) {
      targetdt = Real_t(2.0) * domain.deltatime() / Real_t(3.0) ;
   }

   if (targetdt < domain.deltatime()) {
      domain.deltatime() = targetdt ;
   }

   domain.time() += domain.deltatime() ;

   ++domain.cycle() ;
}

void Lulesh::InitStressTermsForElems(Index_t numElem, 
      Real_t *sigxx, Real_t *sigyy, Real_t *sigzz,
      Real_t *sigxy, Real_t *sigxz, Real_t *sigyz)
{
   //
   // pull in the stresses appropriate to the hydro integration
   //
   for (Index_t i = 0 ; i < numElem ; ++i){
      sigxx[i] =  domain.sx(i)                - domain.p(i) - domain.q(i) ;
      sigyy[i] =                 domain.sy(i) - domain.p(i) - domain.q(i) ;
      sigzz[i] = -domain.sx(i) - domain.sy(i) - domain.p(i) - domain.q(i) ;
      sigxy[i] = domain.txy(i) ;
      sigyz[i] = domain.tyz(i) ;
      sigxz[i] = domain.txz(i) ;
   }
}

void Lulesh::CalcElemShapeFunctionDerivatives( const Real_t* const x,
      const Real_t* const y,
      const Real_t* const z,
      Real_t b[][8],
      Real_t* const volume )
{
   const Real_t x0 = x[0] ;   const Real_t x1 = x[1] ;
   const Real_t x2 = x[2] ;   const Real_t x3 = x[3] ;
   const Real_t x4 = x[4] ;   const Real_t x5 = x[5] ;
   const Real_t x6 = x[6] ;   const Real_t x7 = x[7] ;

   const Real_t y0 = y[0] ;   const Real_t y1 = y[1] ;
   const Real_t y2 = y[2] ;   const Real_t y3 = y[3] ;
   const Real_t y4 = y[4] ;   const Real_t y5 = y[5] ;
   const Real_t y6 = y[6] ;   const Real_t y7 = y[7] ;

   const Real_t z0 = z[0] ;   const Real_t z1 = z[1] ;
   const Real_t z2 = z[2] ;   const Real_t z3 = z[3] ;
   const Real_t z4 = z[4] ;   const Real_t z5 = z[5] ;
   const Real_t z6 = z[6] ;   const Real_t z7 = z[7] ;

   Real_t fjxxi, fjxet, fjxze;
   Real_t fjyxi, fjyet, fjyze;
   Real_t fjzxi, fjzet, fjzze;
   Real_t cjxxi, cjxet, cjxze;
   Real_t cjyxi, cjyet, cjyze;
   Real_t cjzxi, cjzet, cjzze;

   fjxxi = .125 * ( (x6-x0) + (x5-x3) - (x7-x1) - (x4-x2) );
   fjxet = .125 * ( (x6-x0) - (x5-x3) + (x7-x1) - (x4-x2) );
   fjxze = .125 * ( (x6-x0) + (x5-x3) + (x7-x1) + (x4-x2) );

   fjyxi = .125 * ( (y6-y0) + (y5-y3) - (y7-y1) - (y4-y2) );
   fjyet = .125 * ( (y6-y0) - (y5-y3) + (y7-y1) - (y4-y2) );
   fjyze = .125 * ( (y6-y0) + (y5-y3) + (y7-y1) + (y4-y2) );

   fjzxi = .125 * ( (z6-z0) + (z5-z3) - (z7-z1) - (z4-z2) );
   fjzet = .125 * ( (z6-z0) - (z5-z3) + (z7-z1) - (z4-z2) );
   fjzze = .125 * ( (z6-z0) + (z5-z3) + (z7-z1) + (z4-z2) );

   /* compute cofactors */
   cjxxi =    (fjyet * fjzze) - (fjzet * fjyze);
   cjxet =  - (fjyxi * fjzze) + (fjzxi * fjyze);
   cjxze =    (fjyxi * fjzet) - (fjzxi * fjyet);

   cjyxi =  - (fjxet * fjzze) + (fjzet * fjxze);
   cjyet =    (fjxxi * fjzze) - (fjzxi * fjxze);
   cjyze =  - (fjxxi * fjzet) + (fjzxi * fjxet);

   cjzxi =    (fjxet * fjyze) - (fjyet * fjxze);
   cjzet =  - (fjxxi * fjyze) + (fjyxi * fjxze);
   cjzze =    (fjxxi * fjyet) - (fjyxi * fjxet);

   /* calculate partials :
      this need only be done for l = 0,1,2,3   since , by symmetry ,
      (6,7,4,5) = - (0,1,2,3) .
      */
   b[0][0] =   -  cjxxi  -  cjxet  -  cjxze;
   b[0][1] =      cjxxi  -  cjxet  -  cjxze;
   b[0][2] =      cjxxi  +  cjxet  -  cjxze;
   b[0][3] =   -  cjxxi  +  cjxet  -  cjxze;
   b[0][4] = -b[0][2];
   b[0][5] = -b[0][3];
   b[0][6] = -b[0][0];
   b[0][7] = -b[0][1];

   b[1][0] =   -  cjyxi  -  cjyet  -  cjyze;
   b[1][1] =      cjyxi  -  cjyet  -  cjyze;
   b[1][2] =      cjyxi  +  cjyet  -  cjyze;
   b[1][3] =   -  cjyxi  +  cjyet  -  cjyze;
   b[1][4] = -b[1][2];
   b[1][5] = -b[1][3];
   b[1][6] = -b[1][0];
   b[1][7] = -b[1][1];

   b[2][0] =   -  cjzxi  -  cjzet  -  cjzze;
   b[2][1] =      cjzxi  -  cjzet  -  cjzze;
   b[2][2] =      cjzxi  +  cjzet  -  cjzze;
   b[2][3] =   -  cjzxi  +  cjzet  -  cjzze;
   b[2][4] = -b[2][2];
   b[2][5] = -b[2][3];
   b[2][6] = -b[2][0];
   b[2][7] = -b[2][1];

   /* calculate jacobian determinant (volume) */
   *volume = Real_t(8.) * ( fjxet * cjxet + fjyet * cjyet + fjzet * cjzet);
}

void Lulesh::SumElemFaceNormal(Real_t *normalX0, Real_t *normalY0, Real_t *normalZ0,
      Real_t *normalX1, Real_t *normalY1, Real_t *normalZ1,
      Real_t *normalX2, Real_t *normalY2, Real_t *normalZ2,
      Real_t *normalX3, Real_t *normalY3, Real_t *normalZ3,
      const Real_t x0, const Real_t y0, const Real_t z0,
      const Real_t x1, const Real_t y1, const Real_t z1,
      const Real_t x2, const Real_t y2, const Real_t z2,
      const Real_t x3, const Real_t y3, const Real_t z3)
{
   Real_t bisectX0 = Real_t(0.5) * (x3 + x2 - x1 - x0);
   Real_t bisectY0 = Real_t(0.5) * (y3 + y2 - y1 - y0);
   Real_t bisectZ0 = Real_t(0.5) * (z3 + z2 - z1 - z0);
   Real_t bisectX1 = Real_t(0.5) * (x2 + x1 - x3 - x0);
   Real_t bisectY1 = Real_t(0.5) * (y2 + y1 - y3 - y0);
   Real_t bisectZ1 = Real_t(0.5) * (z2 + z1 - z3 - z0);
   Real_t areaX = Real_t(0.25) * (bisectY0 * bisectZ1 - bisectZ0 * bisectY1);
   Real_t areaY = Real_t(0.25) * (bisectZ0 * bisectX1 - bisectX0 * bisectZ1);
   Real_t areaZ = Real_t(0.25) * (bisectX0 * bisectY1 - bisectY0 * bisectX1);

   *normalX0 += areaX;
   *normalX1 += areaX;
   *normalX2 += areaX;
   *normalX3 += areaX;

   *normalY0 += areaY;
   *normalY1 += areaY;
   *normalY2 += areaY;
   *normalY3 += areaY;

   *normalZ0 += areaZ;
   *normalZ1 += areaZ;
   *normalZ2 += areaZ;
   *normalZ3 += areaZ;
}

void Lulesh::CalcElemNodeNormals(Real_t pfx[8],
      Real_t pfy[8],
      Real_t pfz[8],
      const Real_t x[8],
      const Real_t y[8],
      const Real_t z[8])
{
   for (Index_t i = 0 ; i < 8 ; ++i) {
      pfx[i] = Real_t(0.0);
      pfy[i] = Real_t(0.0);
      pfz[i] = Real_t(0.0);
   }
   /* evaluate face one: nodes 0, 1, 2, 3 */
   SumElemFaceNormal(&pfx[0], &pfy[0], &pfz[0],
         &pfx[1], &pfy[1], &pfz[1],
         &pfx[2], &pfy[2], &pfz[2],
         &pfx[3], &pfy[3], &pfz[3],
         x[0], y[0], z[0], x[1], y[1], z[1],
         x[2], y[2], z[2], x[3], y[3], z[3]);
   /* evaluate face two: nodes 0, 4, 5, 1 */
   SumElemFaceNormal(&pfx[0], &pfy[0], &pfz[0],
         &pfx[4], &pfy[4], &pfz[4],
         &pfx[5], &pfy[5], &pfz[5],
         &pfx[1], &pfy[1], &pfz[1],
         x[0], y[0], z[0], x[4], y[4], z[4],
         x[5], y[5], z[5], x[1], y[1], z[1]);
   /* evaluate face three: nodes 1, 5, 6, 2 */
   SumElemFaceNormal(&pfx[1], &pfy[1], &pfz[1],
         &pfx[5], &pfy[5], &pfz[5],
         &pfx[6], &pfy[6], &pfz[6],
         &pfx[2], &pfy[2], &pfz[2],
         x[1], y[1], z[1], x[5], y[5], z[5],
         x[6], y[6], z[6], x[2], y[2], z[2]);
   /* evaluate face four: nodes 2, 6, 7, 3 */
   SumElemFaceNormal(&pfx[2], &pfy[2], &pfz[2],
         &pfx[6], &pfy[6], &pfz[6],
         &pfx[7], &pfy[7], &pfz[7],
         &pfx[3], &pfy[3], &pfz[3],
         x[2], y[2], z[2], x[6], y[6], z[6],
         x[7], y[7], z[7], x[3], y[3], z[3]);
   /* evaluate face five: nodes 3, 7, 4, 0 */
   SumElemFaceNormal(&pfx[3], &pfy[3], &pfz[3],
         &pfx[7], &pfy[7], &pfz[7],
         &pfx[4], &pfy[4], &pfz[4],
         &pfx[0], &pfy[0], &pfz[0],
         x[3], y[3], z[3], x[7], y[7], z[7],
         x[4], y[4], z[4], x[0], y[0], z[0]);
   /* evaluate face six: nodes 4, 7, 6, 5 */
   SumElemFaceNormal(&pfx[4], &pfy[4], &pfz[4],
         &pfx[7], &pfy[7], &pfz[7],
         &pfx[6], &pfy[6], &pfz[6],
         &pfx[5], &pfy[5], &pfz[5],
         x[4], y[4], z[4], x[7], y[7], z[7],
         x[6], y[6], z[6], x[5], y[5], z[5]);
}

void Lulesh::SumElemStressesToNodeForces( const Real_t B[][8],
      const Real_t stress_xx,
      const Real_t stress_yy,
      const Real_t stress_zz,
      const Real_t stress_xy,
      const Real_t stress_xz,
      const Real_t stress_yz,
      Real_t* const fx,
      Real_t* const fy,
      Real_t* const fz )
{
   Real_t pfx0 = B[0][0] ;   Real_t pfx1 = B[0][1] ;
   Real_t pfx2 = B[0][2] ;   Real_t pfx3 = B[0][3] ;
   Real_t pfx4 = B[0][4] ;   Real_t pfx5 = B[0][5] ;
   Real_t pfx6 = B[0][6] ;   Real_t pfx7 = B[0][7] ;

   Real_t pfy0 = B[1][0] ;   Real_t pfy1 = B[1][1] ;
   Real_t pfy2 = B[1][2] ;   Real_t pfy3 = B[1][3] ;
   Real_t pfy4 = B[1][4] ;   Real_t pfy5 = B[1][5] ;
   Real_t pfy6 = B[1][6] ;   Real_t pfy7 = B[1][7] ;

   Real_t pfz0 = B[2][0] ;   Real_t pfz1 = B[2][1] ;
   Real_t pfz2 = B[2][2] ;   Real_t pfz3 = B[2][3] ;
   Real_t pfz4 = B[2][4] ;   Real_t pfz5 = B[2][5] ;
   Real_t pfz6 = B[2][6] ;   Real_t pfz7 = B[2][7] ;

   fx[0] = -( (stress_xx * pfx0) + (stress_xy * pfy0) + (stress_xz * pfz0) );
   fx[1] = -( (stress_xx * pfx1) + (stress_xy * pfy1) + (stress_xz * pfz1) );
   fx[2] = -( (stress_xx * pfx2) + (stress_xy * pfy2) + (stress_xz * pfz2) );
   fx[3] = -( (stress_xx * pfx3) + (stress_xy * pfy3) + (stress_xz * pfz3) );
   fx[4] = -( (stress_xx * pfx4) + (stress_xy * pfy4) + (stress_xz * pfz4) );
   fx[5] = -( (stress_xx * pfx5) + (stress_xy * pfy5) + (stress_xz * pfz5) );
   fx[6] = -( (stress_xx * pfx6) + (stress_xy * pfy6) + (stress_xz * pfz6) );
   fx[7] = -( (stress_xx * pfx7) + (stress_xy * pfy7) + (stress_xz * pfz7) );

   fy[0] = -( (stress_xy * pfx0) + (stress_yy * pfy0) + (stress_yz * pfz0) );
   fy[1] = -( (stress_xy * pfx1) + (stress_yy * pfy1) + (stress_yz * pfz1) );
   fy[2] = -( (stress_xy * pfx2) + (stress_yy * pfy2) + (stress_yz * pfz2) );
   fy[3] = -( (stress_xy * pfx3) + (stress_yy * pfy3) + (stress_yz * pfz3) );
   fy[4] = -( (stress_xy * pfx4) + (stress_yy * pfy4) + (stress_yz * pfz4) );
   fy[5] = -( (stress_xy * pfx5) + (stress_yy * pfy5) + (stress_yz * pfz5) );
   fy[6] = -( (stress_xy * pfx6) + (stress_yy * pfy6) + (stress_yz * pfz6) );
   fy[7] = -( (stress_xy * pfx7) + (stress_yy * pfy7) + (stress_yz * pfz7) );

   fz[0] = -( (stress_xz * pfx0) + (stress_yz * pfy0) + (stress_zz * pfz0) );
   fz[1] = -( (stress_xz * pfx1) + (stress_yz * pfy1) + (stress_zz * pfz1) );
   fz[2] = -( (stress_xz * pfx2) + (stress_yz * pfy2) + (stress_zz * pfz2) );
   fz[3] = -( (stress_xz * pfx3) + (stress_yz * pfy3) + (stress_zz * pfz3) );
   fz[4] = -( (stress_xz * pfx4) + (stress_yz * pfy4) + (stress_zz * pfz4) );
   fz[5] = -( (stress_xz * pfx5) + (stress_yz * pfy5) + (stress_zz * pfz5) );
   fz[6] = -( (stress_xz * pfx6) + (stress_yz * pfy6) + (stress_zz * pfz6) );
   fz[7] = -( (stress_xz * pfx7) + (stress_yz * pfy7) + (stress_zz * pfz7) );
}

void Lulesh::IntegrateStressForElems( Index_t numElem,
      Real_t *sigxx, Real_t *sigyy, Real_t *sigzz,
      Real_t *sigxy, Real_t *sigxz, Real_t *sigyz,
      Real_t *determ)
{
   Real_t B[3][8] ;// shape function derivatives
   Real_t x_local[8] ;
   Real_t y_local[8] ;
   Real_t z_local[8] ;
   Real_t fx_local[8] ;
   Real_t fy_local[8] ;
   Real_t fz_local[8] ;

   // loop over all elements
   for( Index_t k=0 ; k<numElem ; ++k )
   {
      const Index_t* const elemNodes = domain.nodelist(k);

      // get nodal coordinates from global arrays and copy into local arrays.
      for( Index_t lnode=0 ; lnode<8 ; ++lnode )
      {
         Index_t gnode = elemNodes[lnode];
         x_local[lnode] = domain.x(gnode);
         y_local[lnode] = domain.y(gnode);
         z_local[lnode] = domain.z(gnode);
      }

      /* Volume calculation involves extra work for numerical consistency. */
      CalcElemShapeFunctionDerivatives(x_local, y_local, z_local,
            B, &determ[k]);

      CalcElemNodeNormals( B[0] , B[1], B[2],
            x_local, y_local, z_local );

      SumElemStressesToNodeForces( B, sigxx[k], sigyy[k], sigzz[k],
            sigxy[k], sigxz[k], sigyz[k],
            fx_local, fy_local, fz_local ) ;

      // copy nodal force contributions to global force arrray.
      for( Index_t lnode=0 ; lnode<8 ; ++lnode )
      {
         Index_t gnode = elemNodes[lnode];
         domain.fx(gnode) += fx_local[lnode];
         domain.fy(gnode) += fy_local[lnode];
         domain.fz(gnode) += fz_local[lnode];
      }
   }
}

void Lulesh::CollectDomainNodesToElemNodes(const Index_t* elemToNode,
      Real_t elemX[8],
      Real_t elemY[8],
      Real_t elemZ[8])
{
   Index_t nd0i = elemToNode[0] ;
   Index_t nd1i = elemToNode[1] ;
   Index_t nd2i = elemToNode[2] ;
   Index_t nd3i = elemToNode[3] ;
   Index_t nd4i = elemToNode[4] ;
   Index_t nd5i = elemToNode[5] ;
   Index_t nd6i = elemToNode[6] ;
   Index_t nd7i = elemToNode[7] ;

   elemX[0] = domain.x(nd0i);
   elemX[1] = domain.x(nd1i);
   elemX[2] = domain.x(nd2i);
   elemX[3] = domain.x(nd3i);
   elemX[4] = domain.x(nd4i);
   elemX[5] = domain.x(nd5i);
   elemX[6] = domain.x(nd6i);
   elemX[7] = domain.x(nd7i);

   elemY[0] = domain.y(nd0i);
   elemY[1] = domain.y(nd1i);
   elemY[2] = domain.y(nd2i);
   elemY[3] = domain.y(nd3i);
   elemY[4] = domain.y(nd4i);
   elemY[5] = domain.y(nd5i);
   elemY[6] = domain.y(nd6i);
   elemY[7] = domain.y(nd7i);

   elemZ[0] = domain.z(nd0i);
   elemZ[1] = domain.z(nd1i);
   elemZ[2] = domain.z(nd2i);
   elemZ[3] = domain.z(nd3i);
   elemZ[4] = domain.z(nd4i);
   elemZ[5] = domain.z(nd5i);
   elemZ[6] = domain.z(nd6i);
   elemZ[7] = domain.z(nd7i);

}

void Lulesh::VoluDer(const Real_t x0, const Real_t x1, const Real_t x2,
      const Real_t x3, const Real_t x4, const Real_t x5,
      const Real_t y0, const Real_t y1, const Real_t y2,
      const Real_t y3, const Real_t y4, const Real_t y5,
      const Real_t z0, const Real_t z1, const Real_t z2,
      const Real_t z3, const Real_t z4, const Real_t z5,
      Real_t* dvdx, Real_t* dvdy, Real_t* dvdz)
{
   const Real_t twelfth = Real_t(1.0) / Real_t(12.0) ;

   *dvdx =
      (y1 + y2) * (z0 + z1) - (y0 + y1) * (z1 + z2) +
      (y0 + y4) * (z3 + z4) - (y3 + y4) * (z0 + z4) -
      (y2 + y5) * (z3 + z5) + (y3 + y5) * (z2 + z5);
   *dvdy =
      - (x1 + x2) * (z0 + z1) + (x0 + x1) * (z1 + z2) -
      (x0 + x4) * (z3 + z4) + (x3 + x4) * (z0 + z4) +
      (x2 + x5) * (z3 + z5) - (x3 + x5) * (z2 + z5);

   *dvdz =
      - (y1 + y2) * (x0 + x1) + (y0 + y1) * (x1 + x2) -
      (y0 + y4) * (x3 + x4) + (y3 + y4) * (x0 + x4) +
      (y2 + y5) * (x3 + x5) - (y3 + y5) * (x2 + x5);

   *dvdx *= twelfth;
   *dvdy *= twelfth;
   *dvdz *= twelfth;
}

void Lulesh::CalcElemVolumeDerivative(Real_t dvdx[8],
      Real_t dvdy[8],
      Real_t dvdz[8],
      const Real_t x[8],
      const Real_t y[8],
      const Real_t z[8])
{
   VoluDer(x[1], x[2], x[3], x[4], x[5], x[7],
         y[1], y[2], y[3], y[4], y[5], y[7],
         z[1], z[2], z[3], z[4], z[5], z[7],
         &dvdx[0], &dvdy[0], &dvdz[0]);
   VoluDer(x[0], x[1], x[2], x[7], x[4], x[6],
         y[0], y[1], y[2], y[7], y[4], y[6],
         z[0], z[1], z[2], z[7], z[4], z[6],
         &dvdx[3], &dvdy[3], &dvdz[3]);
   VoluDer(x[3], x[0], x[1], x[6], x[7], x[5],
         y[3], y[0], y[1], y[6], y[7], y[5],
         z[3], z[0], z[1], z[6], z[7], z[5],
         &dvdx[2], &dvdy[2], &dvdz[2]);
   VoluDer(x[2], x[3], x[0], x[5], x[6], x[4],
         y[2], y[3], y[0], y[5], y[6], y[4],
         z[2], z[3], z[0], z[5], z[6], z[4],
         &dvdx[1], &dvdy[1], &dvdz[1]);
   VoluDer(x[7], x[6], x[5], x[0], x[3], x[1],
         y[7], y[6], y[5], y[0], y[3], y[1],
         z[7], z[6], z[5], z[0], z[3], z[1],
         &dvdx[4], &dvdy[4], &dvdz[4]);
   VoluDer(x[4], x[7], x[6], x[1], x[0], x[2],
         y[4], y[7], y[6], y[1], y[0], y[2],
         z[4], z[7], z[6], z[1], z[0], z[2],
         &dvdx[5], &dvdy[5], &dvdz[5]);
   VoluDer(x[5], x[4], x[7], x[2], x[1], x[3],
         y[5], y[4], y[7], y[2], y[1], y[3],
         z[5], z[4], z[7], z[2], z[1], z[3],
         &dvdx[6], &dvdy[6], &dvdz[6]);
   VoluDer(x[6], x[5], x[4], x[3], x[2], x[0],
         y[6], y[5], y[4], y[3], y[2], y[0],
         z[6], z[5], z[4], z[3], z[2], z[0],
         &dvdx[7], &dvdy[7], &dvdz[7]);
}

void Lulesh::CalcElemFBHourglassForce(Real_t *xd, Real_t *yd, Real_t *zd,  Real_t *hourgam0,
      Real_t *hourgam1, Real_t *hourgam2, Real_t *hourgam3,
      Real_t *hourgam4, Real_t *hourgam5, Real_t *hourgam6,
      Real_t *hourgam7, Real_t coefficient,
      Real_t *hgfx, Real_t *hgfy, Real_t *hgfz )
{
   Index_t i00=0;
   Index_t i01=1;
   Index_t i02=2;
   Index_t i03=3;

   Real_t h00 =
      hourgam0[i00] * xd[0] + hourgam1[i00] * xd[1] +
      hourgam2[i00] * xd[2] + hourgam3[i00] * xd[3] +
      hourgam4[i00] * xd[4] + hourgam5[i00] * xd[5] +
      hourgam6[i00] * xd[6] + hourgam7[i00] * xd[7];

   Real_t h01 =
      hourgam0[i01] * xd[0] + hourgam1[i01] * xd[1] +
      hourgam2[i01] * xd[2] + hourgam3[i01] * xd[3] +
      hourgam4[i01] * xd[4] + hourgam5[i01] * xd[5] +
      hourgam6[i01] * xd[6] + hourgam7[i01] * xd[7];

   Real_t h02 =
      hourgam0[i02] * xd[0] + hourgam1[i02] * xd[1]+
      hourgam2[i02] * xd[2] + hourgam3[i02] * xd[3]+
      hourgam4[i02] * xd[4] + hourgam5[i02] * xd[5]+
      hourgam6[i02] * xd[6] + hourgam7[i02] * xd[7];

   Real_t h03 =
      hourgam0[i03] * xd[0] + hourgam1[i03] * xd[1] +
      hourgam2[i03] * xd[2] + hourgam3[i03] * xd[3] +
      hourgam4[i03] * xd[4] + hourgam5[i03] * xd[5] +
      hourgam6[i03] * xd[6] + hourgam7[i03] * xd[7];

   hgfx[0] = coefficient *
      (hourgam0[i00] * h00 + hourgam0[i01] * h01 +
       hourgam0[i02] * h02 + hourgam0[i03] * h03);

   hgfx[1] = coefficient *
      (hourgam1[i00] * h00 + hourgam1[i01] * h01 +
       hourgam1[i02] * h02 + hourgam1[i03] * h03);

   hgfx[2] = coefficient *
      (hourgam2[i00] * h00 + hourgam2[i01] * h01 +
       hourgam2[i02] * h02 + hourgam2[i03] * h03);

   hgfx[3] = coefficient *
      (hourgam3[i00] * h00 + hourgam3[i01] * h01 +
       hourgam3[i02] * h02 + hourgam3[i03] * h03);

   hgfx[4] = coefficient *
      (hourgam4[i00] * h00 + hourgam4[i01] * h01 +
       hourgam4[i02] * h02 + hourgam4[i03] * h03);

   hgfx[5] = coefficient *
      (hourgam5[i00] * h00 + hourgam5[i01] * h01 +
       hourgam5[i02] * h02 + hourgam5[i03] * h03);

   hgfx[6] = coefficient *
      (hourgam6[i00] * h00 + hourgam6[i01] * h01 +
       hourgam6[i02] * h02 + hourgam6[i03] * h03);

   hgfx[7] = coefficient *
      (hourgam7[i00] * h00 + hourgam7[i01] * h01 +
       hourgam7[i02] * h02 + hourgam7[i03] * h03);

   h00 =
      hourgam0[i00] * yd[0] + hourgam1[i00] * yd[1] +
      hourgam2[i00] * yd[2] + hourgam3[i00] * yd[3] +
      hourgam4[i00] * yd[4] + hourgam5[i00] * yd[5] +
      hourgam6[i00] * yd[6] + hourgam7[i00] * yd[7];

   h01 =
      hourgam0[i01] * yd[0] + hourgam1[i01] * yd[1] +
      hourgam2[i01] * yd[2] + hourgam3[i01] * yd[3] +
      hourgam4[i01] * yd[4] + hourgam5[i01] * yd[5] +
      hourgam6[i01] * yd[6] + hourgam7[i01] * yd[7];

   h02 =
      hourgam0[i02] * yd[0] + hourgam1[i02] * yd[1]+
      hourgam2[i02] * yd[2] + hourgam3[i02] * yd[3]+
      hourgam4[i02] * yd[4] + hourgam5[i02] * yd[5]+
      hourgam6[i02] * yd[6] + hourgam7[i02] * yd[7];

   h03 =
      hourgam0[i03] * yd[0] + hourgam1[i03] * yd[1] +
      hourgam2[i03] * yd[2] + hourgam3[i03] * yd[3] +
      hourgam4[i03] * yd[4] + hourgam5[i03] * yd[5] +
      hourgam6[i03] * yd[6] + hourgam7[i03] * yd[7];


   hgfy[0] = coefficient *
      (hourgam0[i00] * h00 + hourgam0[i01] * h01 +
       hourgam0[i02] * h02 + hourgam0[i03] * h03);

   hgfy[1] = coefficient *
      (hourgam1[i00] * h00 + hourgam1[i01] * h01 +
       hourgam1[i02] * h02 + hourgam1[i03] * h03);

   hgfy[2] = coefficient *
      (hourgam2[i00] * h00 + hourgam2[i01] * h01 +
       hourgam2[i02] * h02 + hourgam2[i03] * h03);

   hgfy[3] = coefficient *
      (hourgam3[i00] * h00 + hourgam3[i01] * h01 +
       hourgam3[i02] * h02 + hourgam3[i03] * h03);

   hgfy[4] = coefficient *
      (hourgam4[i00] * h00 + hourgam4[i01] * h01 +
       hourgam4[i02] * h02 + hourgam4[i03] * h03);

   hgfy[5] = coefficient *
      (hourgam5[i00] * h00 + hourgam5[i01] * h01 +
       hourgam5[i02] * h02 + hourgam5[i03] * h03);

   hgfy[6] = coefficient *
      (hourgam6[i00] * h00 + hourgam6[i01] * h01 +
       hourgam6[i02] * h02 + hourgam6[i03] * h03);

   hgfy[7] = coefficient *
      (hourgam7[i00] * h00 + hourgam7[i01] * h01 +
       hourgam7[i02] * h02 + hourgam7[i03] * h03);

   h00 =
      hourgam0[i00] * zd[0] + hourgam1[i00] * zd[1] +
      hourgam2[i00] * zd[2] + hourgam3[i00] * zd[3] +
      hourgam4[i00] * zd[4] + hourgam5[i00] * zd[5] +
      hourgam6[i00] * zd[6] + hourgam7[i00] * zd[7];

   h01 =
      hourgam0[i01] * zd[0] + hourgam1[i01] * zd[1] +
      hourgam2[i01] * zd[2] + hourgam3[i01] * zd[3] +
      hourgam4[i01] * zd[4] + hourgam5[i01] * zd[5] +
      hourgam6[i01] * zd[6] + hourgam7[i01] * zd[7];

   h02 =
      hourgam0[i02] * zd[0] + hourgam1[i02] * zd[1]+
      hourgam2[i02] * zd[2] + hourgam3[i02] * zd[3]+
      hourgam4[i02] * zd[4] + hourgam5[i02] * zd[5]+
      hourgam6[i02] * zd[6] + hourgam7[i02] * zd[7];

   h03 =
      hourgam0[i03] * zd[0] + hourgam1[i03] * zd[1] +
      hourgam2[i03] * zd[2] + hourgam3[i03] * zd[3] +
      hourgam4[i03] * zd[4] + hourgam5[i03] * zd[5] +
      hourgam6[i03] * zd[6] + hourgam7[i03] * zd[7];


   hgfz[0] = coefficient *
      (hourgam0[i00] * h00 + hourgam0[i01] * h01 +
       hourgam0[i02] * h02 + hourgam0[i03] * h03);

   hgfz[1] = coefficient *
      (hourgam1[i00] * h00 + hourgam1[i01] * h01 +
       hourgam1[i02] * h02 + hourgam1[i03] * h03);

   hgfz[2] = coefficient *
      (hourgam2[i00] * h00 + hourgam2[i01] * h01 +
       hourgam2[i02] * h02 + hourgam2[i03] * h03);

   hgfz[3] = coefficient *
      (hourgam3[i00] * h00 + hourgam3[i01] * h01 +
       hourgam3[i02] * h02 + hourgam3[i03] * h03);

   hgfz[4] = coefficient *
      (hourgam4[i00] * h00 + hourgam4[i01] * h01 +
       hourgam4[i02] * h02 + hourgam4[i03] * h03);

   hgfz[5] = coefficient *
      (hourgam5[i00] * h00 + hourgam5[i01] * h01 +
       hourgam5[i02] * h02 + hourgam5[i03] * h03);

   hgfz[6] = coefficient *
      (hourgam6[i00] * h00 + hourgam6[i01] * h01 +
       hourgam6[i02] * h02 + hourgam6[i03] * h03);

   hgfz[7] = coefficient *
      (hourgam7[i00] * h00 + hourgam7[i01] * h01 +
       hourgam7[i02] * h02 + hourgam7[i03] * h03);
}

void Lulesh::CalcFBHourglassForceForElems(Real_t *determ,
      Real_t *x8n,      Real_t *y8n,      Real_t *z8n,
      Real_t *dvdx,     Real_t *dvdy,     Real_t *dvdz,
      Real_t hourg)
{
   /*************************************************
    *
    *     FUNCTION: Calculates the Flanagan-Belytschko anti-hourglass
    *               force.
    *
    *************************************************/

   Index_t numElem = domain.numElem() ;

   Real_t hgfx[8], hgfy[8], hgfz[8] ;

   Real_t coefficient;

   Real_t  gamma[4][8];
   Real_t hourgam0[4], hourgam1[4], hourgam2[4], hourgam3[4] ;
   Real_t hourgam4[4], hourgam5[4], hourgam6[4], hourgam7[4];
   Real_t xd1[8], yd1[8], zd1[8] ;

   gamma[0][0] = Real_t( 1.);
   gamma[0][1] = Real_t( 1.);
   gamma[0][2] = Real_t(-1.);
   gamma[0][3] = Real_t(-1.);
   gamma[0][4] = Real_t(-1.);
   gamma[0][5] = Real_t(-1.);
   gamma[0][6] = Real_t( 1.);
   gamma[0][7] = Real_t( 1.);
   gamma[1][0] = Real_t( 1.);
   gamma[1][1] = Real_t(-1.);
   gamma[1][2] = Real_t(-1.);
   gamma[1][3] = Real_t( 1.);
   gamma[1][4] = Real_t(-1.);
   gamma[1][5] = Real_t( 1.);
   gamma[1][6] = Real_t( 1.);
   gamma[1][7] = Real_t(-1.);
   gamma[2][0] = Real_t( 1.);
   gamma[2][1] = Real_t(-1.);
   gamma[2][2] = Real_t( 1.);
   gamma[2][3] = Real_t(-1.);
   gamma[2][4] = Real_t( 1.);
   gamma[2][5] = Real_t(-1.);
   gamma[2][6] = Real_t( 1.);
   gamma[2][7] = Real_t(-1.);
   gamma[3][0] = Real_t(-1.);
   gamma[3][1] = Real_t( 1.);
   gamma[3][2] = Real_t(-1.);
   gamma[3][3] = Real_t( 1.);
   gamma[3][4] = Real_t( 1.);
   gamma[3][5] = Real_t(-1.);
   gamma[3][6] = Real_t( 1.);
   gamma[3][7] = Real_t(-1.);

   /*************************************************/
   /*    compute the hourglass modes */


   for(Index_t i2=0;i2<numElem;++i2){
      const Index_t *elemToNode = domain.nodelist(i2);
      Index_t i3=8*i2;
      Real_t volinv=Real_t(1.0)/determ[i2];
      Real_t ss1, mass1, crqt, volume13 ;
      for(Index_t i1=0;i1<4;++i1){

         Real_t hourmodx =
            x8n[i3] * gamma[i1][0] + x8n[i3+1] * gamma[i1][1] +
            x8n[i3+2] * gamma[i1][2] + x8n[i3+3] * gamma[i1][3] +
            x8n[i3+4] * gamma[i1][4] + x8n[i3+5] * gamma[i1][5] +
            x8n[i3+6] * gamma[i1][6] + x8n[i3+7] * gamma[i1][7];

         Real_t hourmody =
            y8n[i3] * gamma[i1][0] + y8n[i3+1] * gamma[i1][1] +
            y8n[i3+2] * gamma[i1][2] + y8n[i3+3] * gamma[i1][3] +
            y8n[i3+4] * gamma[i1][4] + y8n[i3+5] * gamma[i1][5] +
            y8n[i3+6] * gamma[i1][6] + y8n[i3+7] * gamma[i1][7];

         Real_t hourmodz =
            z8n[i3] * gamma[i1][0] + z8n[i3+1] * gamma[i1][1] +
            z8n[i3+2] * gamma[i1][2] + z8n[i3+3] * gamma[i1][3] +
            z8n[i3+4] * gamma[i1][4] + z8n[i3+5] * gamma[i1][5] +
            z8n[i3+6] * gamma[i1][6] + z8n[i3+7] * gamma[i1][7];

         hourgam0[i1] = gamma[i1][0] -  volinv*(dvdx[i3  ] * hourmodx +
               dvdy[i3  ] * hourmody +
               dvdz[i3  ] * hourmodz );

         hourgam1[i1] = gamma[i1][1] -  volinv*(dvdx[i3+1] * hourmodx +
               dvdy[i3+1] * hourmody +
               dvdz[i3+1] * hourmodz );

         hourgam2[i1] = gamma[i1][2] -  volinv*(dvdx[i3+2] * hourmodx +
               dvdy[i3+2] * hourmody +
               dvdz[i3+2] * hourmodz );

         hourgam3[i1] = gamma[i1][3] -  volinv*(dvdx[i3+3] * hourmodx +
               dvdy[i3+3] * hourmody +
               dvdz[i3+3] * hourmodz );

         hourgam4[i1] = gamma[i1][4] -  volinv*(dvdx[i3+4] * hourmodx +
               dvdy[i3+4] * hourmody +
               dvdz[i3+4] * hourmodz );

         hourgam5[i1] = gamma[i1][5] -  volinv*(dvdx[i3+5] * hourmodx +
               dvdy[i3+5] * hourmody +
               dvdz[i3+5] * hourmodz );

         hourgam6[i1] = gamma[i1][6] -  volinv*(dvdx[i3+6] * hourmodx +
               dvdy[i3+6] * hourmody +
               dvdz[i3+6] * hourmodz );

         hourgam7[i1] = gamma[i1][7] -  volinv*(dvdx[i3+7] * hourmodx +
               dvdy[i3+7] * hourmody +
               dvdz[i3+7] * hourmodz );

      }

      /* compute forces */
      /* store forces into h arrays (force arrays) */

      ss1=domain.ss(i2);
      mass1=domain.elemMass(i2);
      volume13=CBRT(determ[i2]);
      crqt = domain.crqt() ;

      Index_t n0si2 = elemToNode[0];
      Index_t n1si2 = elemToNode[1];
      Index_t n2si2 = elemToNode[2];
      Index_t n3si2 = elemToNode[3];
      Index_t n4si2 = elemToNode[4];
      Index_t n5si2 = elemToNode[5];
      Index_t n6si2 = elemToNode[6];
      Index_t n7si2 = elemToNode[7];

      xd1[0] = domain.xd(n0si2);
      xd1[1] = domain.xd(n1si2);
      xd1[2] = domain.xd(n2si2);
      xd1[3] = domain.xd(n3si2);
      xd1[4] = domain.xd(n4si2);
      xd1[5] = domain.xd(n5si2);
      xd1[6] = domain.xd(n6si2);
      xd1[7] = domain.xd(n7si2);

      yd1[0] = domain.yd(n0si2);
      yd1[1] = domain.yd(n1si2);
      yd1[2] = domain.yd(n2si2);
      yd1[3] = domain.yd(n3si2);
      yd1[4] = domain.yd(n4si2);
      yd1[5] = domain.yd(n5si2);
      yd1[6] = domain.yd(n6si2);
      yd1[7] = domain.yd(n7si2);

      zd1[0] = domain.zd(n0si2);
      zd1[1] = domain.zd(n1si2);
      zd1[2] = domain.zd(n2si2);
      zd1[3] = domain.zd(n3si2);
      zd1[4] = domain.zd(n4si2);
      zd1[5] = domain.zd(n5si2);
      zd1[6] = domain.zd(n6si2);
      zd1[7] = domain.zd(n7si2);

      coefficient = - hourg * crqt * ss1 * mass1 / volume13;

      CalcElemFBHourglassForce(xd1,yd1,zd1,
            hourgam0,hourgam1,hourgam2,hourgam3,
            hourgam4,hourgam5,hourgam6,hourgam7,
            coefficient, hgfx, hgfy, hgfz);

      domain.fx(n0si2) += hgfx[0];
      domain.fy(n0si2) += hgfy[0];
      domain.fz(n0si2) += hgfz[0];

      domain.fx(n1si2) += hgfx[1];
      domain.fy(n1si2) += hgfy[1];
      domain.fz(n1si2) += hgfz[1];

      domain.fx(n2si2) += hgfx[2];
      domain.fy(n2si2) += hgfy[2];
      domain.fz(n2si2) += hgfz[2];

      domain.fx(n3si2) += hgfx[3];
      domain.fy(n3si2) += hgfy[3];
      domain.fz(n3si2) += hgfz[3];

      domain.fx(n4si2) += hgfx[4];
      domain.fy(n4si2) += hgfy[4];
      domain.fz(n4si2) += hgfz[4];

      domain.fx(n5si2) += hgfx[5];
      domain.fy(n5si2) += hgfy[5];
      domain.fz(n5si2) += hgfz[5];

      domain.fx(n6si2) += hgfx[6];
      domain.fy(n6si2) += hgfy[6];
      domain.fz(n6si2) += hgfz[6];

      domain.fx(n7si2) += hgfx[7];
      domain.fy(n7si2) += hgfy[7];
      domain.fz(n7si2) += hgfz[7];
   }
}

void Lulesh::CalcHourglassControlForElems(Real_t determ[], Real_t hgcoef)
{
   Index_t i, ii, jj ;
   Real_t  x1[8],  y1[8],  z1[8] ;
   Real_t pfx[8], pfy[8], pfz[8] ;
   Index_t numElem = domain.numElem() ;
   Index_t numElem8 = numElem * 8 ;
   Real_t *dvdx = Allocate<Real_t>(numElem8) ;
   Real_t *dvdy = Allocate<Real_t>(numElem8) ;
   Real_t *dvdz = Allocate<Real_t>(numElem8) ;
   Real_t *x8n  = Allocate<Real_t>(numElem8) ;
   Real_t *y8n  = Allocate<Real_t>(numElem8) ;
   Real_t *z8n  = Allocate<Real_t>(numElem8) ;

   /* start loop over elements */
   for (i=0 ; i<numElem ; ++i){

      Index_t* elemToNode = domain.nodelist(i);
      CollectDomainNodesToElemNodes(elemToNode, x1, y1, z1);

      CalcElemVolumeDerivative(pfx, pfy, pfz, x1, y1, z1);

      /* load into temporary storage for FB Hour Glass control */
      for(ii=0;ii<8;++ii){
         jj=8*i+ii;

         dvdx[jj] = pfx[ii];
         dvdy[jj] = pfy[ii];
         dvdz[jj] = pfz[ii];
         x8n[jj]  = x1[ii];
         y8n[jj]  = y1[ii];
         z8n[jj]  = z1[ii];
      }

      determ[i] = domain.volo(i) * domain.v(i);

      /* Do a check for negative volumes */
      if ( domain.v(i) <= Real_t(0.0) ) {
#if defined(COEVP_MPI)
         MPI_Abort(MPI_COMM_WORLD, VolumeError) ;
#else
         exit(VolumeError) ;
#endif
      }
   }

   if ( hgcoef > Real_t(0.) ) {
      CalcFBHourglassForceForElems(determ,x8n,y8n,z8n,dvdx,dvdy,dvdz,hgcoef) ;
   }

   Release(&z8n) ;
   Release(&y8n) ;
   Release(&x8n) ;
   Release(&dvdz) ;
   Release(&dvdy) ;
   Release(&dvdx) ;

   return ;
}

void Lulesh::CalcVolumeForceForElems()
{
   Index_t numElem = domain.numElem() ;
   if (numElem != 0) {
      Real_t  hgcoef = domain.hgcoef() ;
      Real_t *sigxx  = Allocate<Real_t>(numElem) ;
      Real_t *sigyy  = Allocate<Real_t>(numElem) ;
      Real_t *sigzz  = Allocate<Real_t>(numElem) ;
      Real_t *sigxy  = Allocate<Real_t>(numElem) ;
      Real_t *sigxz  = Allocate<Real_t>(numElem) ;
      Real_t *sigyz  = Allocate<Real_t>(numElem) ;
      Real_t *determ = Allocate<Real_t>(numElem) ;

      /* Sum contributions to total stress tensor */
      InitStressTermsForElems(numElem, sigxx, sigyy, sigzz,
            sigxy, sigxz, sigyz);

      // call elemlib stress integration loop to produce nodal forces from
      // material stresses.
      IntegrateStressForElems( numElem, sigxx, sigyy, sigzz,
            sigxy, sigxz, sigyz, determ) ;

      // check for negative element volume
      for ( Index_t k=0 ; k<numElem ; ++k ) {
         if (determ[k] <= Real_t(0.0)) {
#if defined(COEVP_MPI)
            MPI_Abort(MPI_COMM_WORLD, VolumeError) ;
#else
            exit(VolumeError) ;
#endif
         }
      }

      CalcHourglassControlForElems(determ, hgcoef) ;

      Release(&determ) ;
      Release(&sigyz) ;
      Release(&sigxz) ;
      Release(&sigxy) ;
      Release(&sigzz) ;
      Release(&sigyy) ;
      Release(&sigxx) ;
   }
}

void Lulesh::CalcForceForNodes()
{
   Index_t numNode = domain.numNode() ;
#if defined(COEVP_MPI)
   Real_t *fieldData[3] ;

   CommRecv(&domain, MSG_COMM_SBN, 3, domain.commNodes()) ;
#endif

   for (Index_t i=0; i<numNode; ++i) {
      domain.fx(i) = Real_t(0.0) ;
      domain.fy(i) = Real_t(0.0) ;
      domain.fz(i) = Real_t(0.0) ;
   }

   /* Calcforce calls partial, force, hourq */
   CalcVolumeForceForElems() ;

#if defined(COEVP_MPI)
   fieldData[0] = &domain.fx(0) ;
   fieldData[1] = &domain.fy(0) ;
   fieldData[2] = &domain.fz(0) ;
   CommSend(&domain, MSG_COMM_SBN, 3, fieldData,
         domain.planeNodeIds, domain.commNodes(), domain.sliceHeight()) ;
   CommSBN(&domain, 3, fieldData,
         domain.planeNodeIds, domain.commNodes(), domain.sliceHeight()) ;
#endif

}

void Lulesh::CalcAccelerationForNodes()
{
   Index_t numNode = domain.numNode() ;
   for (Index_t i = 0; i < numNode; ++i) {
      domain.xdd(i) = domain.fx(i) / domain.nodalMass(i);
      domain.ydd(i) = domain.fy(i) / domain.nodalMass(i);
      domain.zdd(i) = domain.fz(i) / domain.nodalMass(i);
   }
}

void Lulesh::ApplyAccelerationBoundaryConditionsForNodes()
{
   Index_t numBoundaryNodes = domain.numSymmNodesBoundary() ;

   Index_t numImpactNodes   = domain.numSymmNodesImpact() ;

#if defined(COEVP_MPI)||defined(__CHARMC__)
  if (domain.sliceLoc() == 0)
#endif
   {
      for(Index_t i=0 ; i<numImpactNodes ; ++i)
         domain.xdd(domain.symmX(i)) = Real_t(0.0) ;
   }

   for(Index_t i=0 ; i<numBoundaryNodes ; ++i)
      domain.ydd(domain.symmY(i)) = Real_t(0.0) ;

   for(Index_t i=0 ; i<numBoundaryNodes ; ++i)
      domain.zdd(domain.symmZ(i)) = Real_t(0.0) ;
}

void Lulesh::CalcVelocityForNodes(const Real_t dt, const Real_t u_cut)
{
   Index_t numNode = domain.numNode() ;

   for ( Index_t i = 0 ; i < numNode ; ++i )
   {
      Real_t xdtmp, ydtmp, zdtmp ;

      xdtmp = domain.xd(i) + domain.xdd(i) * dt ;
      if( FABS(xdtmp) < u_cut ) xdtmp = Real_t(0.0);
      domain.xd(i) = xdtmp ;

      ydtmp = domain.yd(i) + domain.ydd(i) * dt ;
      if( FABS(ydtmp) < u_cut ) ydtmp = Real_t(0.0);
      domain.yd(i) = ydtmp ;

      zdtmp = domain.zd(i) + domain.zdd(i) * dt ;
      if( FABS(zdtmp) < u_cut ) zdtmp = Real_t(0.0);
      domain.zd(i) = zdtmp ;
   }
}

void Lulesh::CalcPositionForNodes(const Real_t dt)
{
   Index_t numNode = domain.numNode() ;

   for ( Index_t i = 0 ; i < numNode ; ++i )
   {
      domain.x(i) += domain.xd(i) * dt ;
      domain.y(i) += domain.yd(i) * dt ;
      domain.z(i) += domain.zd(i) * dt ;
   }
}

void Lulesh::LagrangeNodal1()
{
   /* time of boundary condition evaluation is beginning of step for force and
    * acceleration boundary conditions. */
   CalcForceForNodes();
}

void Lulesh::LagrangeNodal2()
{
   const Real_t delt = domain.deltatime() ;
   Real_t u_cut = domain.u_cut() ;

   CalcAccelerationForNodes();

   ApplyAccelerationBoundaryConditionsForNodes();

   CalcVelocityForNodes( delt, u_cut ) ;

   CalcPositionForNodes( delt );
}


void Lulesh::LagrangeNodal()
{
#if defined(COEVP_MPI) && defined(SEDOV_SYNC_POS_VEL_EARLY)
   Real_t *fieldData[6] ;
#endif

   const Real_t delt = domain.deltatime() ;
   Real_t u_cut = domain.u_cut() ;

   /* time of boundary condition evaluation is beginning of step for force and
    * acceleration boundary conditions. */
   LagrangeNodal1();

#if defined(COEVP_MPI) && defined(SEDOV_SYNC_POS_VEL_EARLY)
   CommRecv(&domain, MSG_SYNC_POS_VEL, 6, domain.commNodes(), false) ;
#endif

   LagrangeNodal2();

#if defined(COEVP_MPI) && defined(SEDOV_SYNC_POS_VEL_EARLY)
   fieldData[0] = &domain.x(0) ;
   fieldData[1] = &domain.y(0) ;
   fieldData[2] = &domain.z(0) ;
   fieldData[3] = &domain.xd(0) ;
   fieldData[4] = &domain.yd(0) ;
   fieldData[5] = &domain.zd(0) ;

   CommSend(&domain, MSG_SYNC_POS_VEL, 6, fieldData,
         domain.planeNodeIds, domain.commNodes(), domain.sliceHeight(),
         false) ;
   CommSyncPosVel(&domain,
         domain.planeNodeIds, domain.commNodes(), domain.sliceHeight()) ;
#endif

   return;
}

   static inline
Real_t CalcElemVolume( const Real_t x0, const Real_t x1,
      const Real_t x2, const Real_t x3,
      const Real_t x4, const Real_t x5,
      const Real_t x6, const Real_t x7,
      const Real_t y0, const Real_t y1,
      const Real_t y2, const Real_t y3,
      const Real_t y4, const Real_t y5,
      const Real_t y6, const Real_t y7,
      const Real_t z0, const Real_t z1,
      const Real_t z2, const Real_t z3,
      const Real_t z4, const Real_t z5,
      const Real_t z6, const Real_t z7 )
{
   Real_t twelveth = Real_t(1.0)/Real_t(12.0);

   Real_t dx61 = x6 - x1;
   Real_t dy61 = y6 - y1;
   Real_t dz61 = z6 - z1;

   Real_t dx70 = x7 - x0;
   Real_t dy70 = y7 - y0;
   Real_t dz70 = z7 - z0;

   Real_t dx63 = x6 - x3;
   Real_t dy63 = y6 - y3;
   Real_t dz63 = z6 - z3;

   Real_t dx20 = x2 - x0;
   Real_t dy20 = y2 - y0;
   Real_t dz20 = z2 - z0;

   Real_t dx50 = x5 - x0;
   Real_t dy50 = y5 - y0;
   Real_t dz50 = z5 - z0;

   Real_t dx64 = x6 - x4;
   Real_t dy64 = y6 - y4;
   Real_t dz64 = z6 - z4;

   Real_t dx31 = x3 - x1;
   Real_t dy31 = y3 - y1;
   Real_t dz31 = z3 - z1;

   Real_t dx72 = x7 - x2;
   Real_t dy72 = y7 - y2;
   Real_t dz72 = z7 - z2;

   Real_t dx43 = x4 - x3;
   Real_t dy43 = y4 - y3;
   Real_t dz43 = z4 - z3;

   Real_t dx57 = x5 - x7;
   Real_t dy57 = y5 - y7;
   Real_t dz57 = z5 - z7;

   Real_t dx14 = x1 - x4;
   Real_t dy14 = y1 - y4;
   Real_t dz14 = z1 - z4;

   Real_t dx25 = x2 - x5;
   Real_t dy25 = y2 - y5;
   Real_t dz25 = z2 - z5;

#define TRIPLE_PRODUCT(x1, y1, z1, x2, y2, z2, x3, y3, z3) \
   ((x1)*((y2)*(z3) - (z2)*(y3)) + (x2)*((z1)*(y3) - (y1)*(z3)) + (x3)*((y1)*(z2) - (z1)*(y2)))

   Real_t volume =
      TRIPLE_PRODUCT(dx31 + dx72, dx63, dx20,
            dy31 + dy72, dy63, dy20,
            dz31 + dz72, dz63, dz20) +
      TRIPLE_PRODUCT(dx43 + dx57, dx64, dx70,
            dy43 + dy57, dy64, dy70,
            dz43 + dz57, dz64, dz70) +
      TRIPLE_PRODUCT(dx14 + dx25, dx61, dx50,
            dy14 + dy25, dy61, dy50,
            dz14 + dz25, dz61, dz50);

#undef TRIPLE_PRODUCT

   volume *= twelveth;

   return volume ;
}

   static inline
Real_t CalcElemVolume( const Real_t x[8], const Real_t y[8], const Real_t z[8] )
{
   return CalcElemVolume( x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7],
         y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7],
         z[0], z[1], z[2], z[3], z[4], z[5], z[6], z[7]);
}

   static inline
Real_t AreaFace( const Real_t x0, const Real_t x1,
      const Real_t x2, const Real_t x3,
      const Real_t y0, const Real_t y1,
      const Real_t y2, const Real_t y3,
      const Real_t z0, const Real_t z1,
      const Real_t z2, const Real_t z3)
{
   Real_t fx = (x2 - x0) - (x3 - x1);
   Real_t fy = (y2 - y0) - (y3 - y1);
   Real_t fz = (z2 - z0) - (z3 - z1);
   Real_t gx = (x2 - x0) + (x3 - x1);
   Real_t gy = (y2 - y0) + (y3 - y1);
   Real_t gz = (z2 - z0) + (z3 - z1);
   Real_t area =
      (fx * fx + fy * fy + fz * fz) *
      (gx * gx + gy * gy + gz * gz) -
      (fx * gx + fy * gy + fz * gz) *
      (fx * gx + fy * gy + fz * gz);
   return area ;
}

   static inline
Real_t CalcElemCharacteristicLength( const Real_t x[8],
      const Real_t y[8],
      const Real_t z[8],
      const Real_t volume)
{
   Real_t a, charLength = Real_t(0.0);

   a = AreaFace(x[0],x[1],x[2],x[3],
         y[0],y[1],y[2],y[3],
         z[0],z[1],z[2],z[3]) ;
   charLength = std::max(a,charLength) ;

   a = AreaFace(x[4],x[5],x[6],x[7],
         y[4],y[5],y[6],y[7],
         z[4],z[5],z[6],z[7]) ;
   charLength = std::max(a,charLength) ;

   a = AreaFace(x[0],x[1],x[5],x[4],
         y[0],y[1],y[5],y[4],
         z[0],z[1],z[5],z[4]) ;
   charLength = std::max(a,charLength) ;

   a = AreaFace(x[1],x[2],x[6],x[5],
         y[1],y[2],y[6],y[5],
         z[1],z[2],z[6],z[5]) ;
   charLength = std::max(a,charLength) ;

   a = AreaFace(x[2],x[3],x[7],x[6],
         y[2],y[3],y[7],y[6],
         z[2],z[3],z[7],z[6]) ;
   charLength = std::max(a,charLength) ;

   a = AreaFace(x[3],x[0],x[4],x[7],
         y[3],y[0],y[4],y[7],
         z[3],z[0],z[4],z[7]) ;
   charLength = std::max(a,charLength) ;

   charLength = Real_t(4.0) * volume / SQRT(charLength);

   return charLength;
}

void Lulesh::CalcElemVelocityGradient( const Real_t* const xvel,
      const Real_t* const yvel,
      const Real_t* const zvel,
      const Real_t b[][8],
      const Real_t detJ,
      Real_t* const d, Real_t* const w )
{
   const Real_t inv_detJ = Real_t(1.0) / detJ ;
   Real_t dyddx, dxddy, dzddx, dxddz, dzddy, dyddz;
   const Real_t* const pfx = b[0];
   const Real_t* const pfy = b[1];
   const Real_t* const pfz = b[2];

   d[0] = inv_detJ * ( pfx[0] * (xvel[0]-xvel[6])
         + pfx[1] * (xvel[1]-xvel[7])
         + pfx[2] * (xvel[2]-xvel[4])
         + pfx[3] * (xvel[3]-xvel[5]) );

   d[1] = inv_detJ * ( pfy[0] * (yvel[0]-yvel[6])
         + pfy[1] * (yvel[1]-yvel[7])
         + pfy[2] * (yvel[2]-yvel[4])
         + pfy[3] * (yvel[3]-yvel[5]) );

   d[2] = inv_detJ * ( pfz[0] * (zvel[0]-zvel[6])
         + pfz[1] * (zvel[1]-zvel[7])
         + pfz[2] * (zvel[2]-zvel[4])
         + pfz[3] * (zvel[3]-zvel[5]) );

   dyddx  = inv_detJ * ( pfx[0] * (yvel[0]-yvel[6])
         + pfx[1] * (yvel[1]-yvel[7])
         + pfx[2] * (yvel[2]-yvel[4])
         + pfx[3] * (yvel[3]-yvel[5]) );

   dxddy  = inv_detJ * ( pfy[0] * (xvel[0]-xvel[6])
         + pfy[1] * (xvel[1]-xvel[7])
         + pfy[2] * (xvel[2]-xvel[4])
         + pfy[3] * (xvel[3]-xvel[5]) );

   dzddx  = inv_detJ * ( pfx[0] * (zvel[0]-zvel[6])
         + pfx[1] * (zvel[1]-zvel[7])
         + pfx[2] * (zvel[2]-zvel[4])
         + pfx[3] * (zvel[3]-zvel[5]) );

   dxddz  = inv_detJ * ( pfz[0] * (xvel[0]-xvel[6])
         + pfz[1] * (xvel[1]-xvel[7])
         + pfz[2] * (xvel[2]-xvel[4])
         + pfz[3] * (xvel[3]-xvel[5]) );

   dzddy  = inv_detJ * ( pfy[0] * (zvel[0]-zvel[6])
         + pfy[1] * (zvel[1]-zvel[7])
         + pfy[2] * (zvel[2]-zvel[4])
         + pfy[3] * (zvel[3]-zvel[5]) );

   dyddz  = inv_detJ * ( pfz[0] * (yvel[0]-yvel[6])
         + pfz[1] * (yvel[1]-yvel[7])
         + pfz[2] * (yvel[2]-yvel[4])
         + pfz[3] * (yvel[3]-yvel[5]) );
   d[5]  = Real_t( .5) * ( dxddy + dyddx );
   d[4]  = Real_t( .5) * ( dxddz + dzddx );
   d[3]  = Real_t( .5) * ( dzddy + dyddz );
   w[2]  = Real_t( .5) * ( dyddx - dxddy );
   w[1]  = Real_t( .5) * ( dxddz - dzddx );
   w[0]  = Real_t( .5) * ( dzddy - dyddz );
}

void Lulesh::CalcKinematicsForElems( Index_t numElem, Real_t dt )
{
   // loop over all elements
#ifdef _OPENMP
#pragma omp parallel for
#endif
   for( Index_t k=0 ; k<numElem ; ++k )
   {
      Real_t B[3][8] ; /** shape function derivatives */
      Real_t D[6] ;
      Real_t W[3] ;
      Real_t x_local[8] ;
      Real_t y_local[8] ;
      Real_t z_local[8] ;
      Real_t xd_local[8] ;
      Real_t yd_local[8] ;
      Real_t zd_local[8] ;
      Real_t detJ = Real_t(0.0) ;

      Real_t volume ;
      Real_t relativeVolume ;
      const Index_t* const elemToNode = domain.nodelist(k) ;

      // get nodal coordinates from global arrays and copy into local arrays.
      for( Index_t lnode=0 ; lnode<8 ; ++lnode )
      {
         Index_t gnode = elemToNode[lnode];
         x_local[lnode] = domain.x(gnode);
         y_local[lnode] = domain.y(gnode);
         z_local[lnode] = domain.z(gnode);
      }

      // volume calculations
      volume = CalcElemVolume(x_local, y_local, z_local );
      relativeVolume = volume / domain.volo(k) ;
      domain.vnew(k) = relativeVolume ;
      domain.delv(k) = relativeVolume - domain.v(k) ;

      // set characteristic length
      domain.arealg(k) = CalcElemCharacteristicLength(x_local,
            y_local,
            z_local,
            volume);

      // get nodal velocities from global array and copy into local arrays.
      for( Index_t lnode=0 ; lnode<8 ; ++lnode )
      {
         Index_t gnode = elemToNode[lnode];
         xd_local[lnode] = domain.xd(gnode);
         yd_local[lnode] = domain.yd(gnode);
         zd_local[lnode] = domain.zd(gnode);
      }

      // compute the velocity gradient at the new time (i.e., before the
      // nodal positions get backed up a half step below).  Question:
      // where are the velocities centered at this point?

      CalcElemShapeFunctionDerivatives( x_local,
            y_local,
            z_local,
            B, &detJ );

      CalcElemVelocityGradient( xd_local,
            yd_local,
            zd_local,
            B, detJ, D, W );

      Tensor2Gen L;

      L(1,1) = D[0];         // dxddx
      L(1,2) = D[5] - W[2];  // dyddx
      L(1,3) = D[4] + W[1];  // dzddx
      L(2,1) = D[5] + W[2];  // dxddy 
      L(2,2) = D[1];         // dyddy
      L(2,3) = D[3] - W[0];  // dzddy
      L(3,1) = D[4] - W[1];  // dxddz
      L(3,2) = D[3] + W[0];  // dyddz
      L(3,3) = D[2];         // dzddz

      domain.cm_vel_grad(k) = L;
      domain.cm_vol_chng(k) = relativeVolume/domain.v(k);

      Real_t dt2 = Real_t(0.5) * dt;
      for ( Index_t j=0 ; j<8 ; ++j )
      {
         x_local[j] -= dt2 * xd_local[j];
         y_local[j] -= dt2 * yd_local[j];
         z_local[j] -= dt2 * zd_local[j];
      }

      CalcElemShapeFunctionDerivatives( x_local,
            y_local,
            z_local,
            B, &detJ );

      CalcElemVelocityGradient( xd_local,
            yd_local,
            zd_local,
            B, detJ, D, W );

      // put velocity gradient quantities into their global arrays.
      domain.dxx(k) = D[0];
      domain.dyy(k) = D[1];
      domain.dzz(k) = D[2];
      domain.dyz(k) = D[3];
      domain.dxz(k) = D[4];
      domain.dxy(k) = D[5];
      domain.wxx(k) = W[0];
      domain.wyy(k) = W[1];
      domain.wzz(k) = W[2];
   }
}

void Lulesh::CalcLagrangeElements(Real_t deltatime)
{
   Index_t numElem = domain.numElem() ;
   if (numElem > 0) {
      CalcKinematicsForElems(numElem, deltatime) ;

      // element loop to do some stuff not included in the elemlib function.
      for ( Index_t k=0 ; k<numElem ; ++k )
      {
         // calc strain rate and apply as constraint (only done in FB element)
         Real_t vdov = domain.dxx(k) + domain.dyy(k) + domain.dzz(k) ;
         Real_t vdovthird = vdov/Real_t(3.0) ;

         // make the rate of deformation tensor deviatoric
         domain.vdov(k) = vdov ;
         domain.dxx(k) -= vdovthird ;
         domain.dyy(k) -= vdovthird ;
         domain.dzz(k) -= vdovthird ;

         // See if any volumes are negative, and take appropriate action.
         if (domain.vnew(k) <= Real_t(0.0))
         {
#if defined(COEVP_MPI)
            MPI_Abort(MPI_COMM_WORLD, VolumeError) ;
#else
            exit(VolumeError) ;
#endif
         }
      }
   }
}

void Lulesh::CalcMonotonicQGradientsForElems()
{
#define SUM4(a,b,c,d) (a + b + c + d)
   Index_t numElem = domain.numElem() ;
   const Real_t ptiny = Real_t(1.e-36) ;

   for (Index_t i = 0 ; i < numElem ; ++i ) {
      Real_t ax,ay,az ;
      Real_t dxv,dyv,dzv ;

      const Index_t *elemToNode = domain.nodelist(i);
      Index_t n0 = elemToNode[0] ;
      Index_t n1 = elemToNode[1] ;
      Index_t n2 = elemToNode[2] ;
      Index_t n3 = elemToNode[3] ;
      Index_t n4 = elemToNode[4] ;
      Index_t n5 = elemToNode[5] ;
      Index_t n6 = elemToNode[6] ;
      Index_t n7 = elemToNode[7] ;

      Real_t x0 = domain.x(n0) ;
      Real_t x1 = domain.x(n1) ;
      Real_t x2 = domain.x(n2) ;
      Real_t x3 = domain.x(n3) ;
      Real_t x4 = domain.x(n4) ;
      Real_t x5 = domain.x(n5) ;
      Real_t x6 = domain.x(n6) ;
      Real_t x7 = domain.x(n7) ;

      Real_t y0 = domain.y(n0) ;
      Real_t y1 = domain.y(n1) ;
      Real_t y2 = domain.y(n2) ;
      Real_t y3 = domain.y(n3) ;
      Real_t y4 = domain.y(n4) ;
      Real_t y5 = domain.y(n5) ;
      Real_t y6 = domain.y(n6) ;
      Real_t y7 = domain.y(n7) ;

      Real_t z0 = domain.z(n0) ;
      Real_t z1 = domain.z(n1) ;
      Real_t z2 = domain.z(n2) ;
      Real_t z3 = domain.z(n3) ;
      Real_t z4 = domain.z(n4) ;
      Real_t z5 = domain.z(n5) ;
      Real_t z6 = domain.z(n6) ;
      Real_t z7 = domain.z(n7) ;

      Real_t xv0 = domain.xd(n0) ;
      Real_t xv1 = domain.xd(n1) ;
      Real_t xv2 = domain.xd(n2) ;
      Real_t xv3 = domain.xd(n3) ;
      Real_t xv4 = domain.xd(n4) ;
      Real_t xv5 = domain.xd(n5) ;
      Real_t xv6 = domain.xd(n6) ;
      Real_t xv7 = domain.xd(n7) ;

      Real_t yv0 = domain.yd(n0) ;
      Real_t yv1 = domain.yd(n1) ;
      Real_t yv2 = domain.yd(n2) ;
      Real_t yv3 = domain.yd(n3) ;
      Real_t yv4 = domain.yd(n4) ;
      Real_t yv5 = domain.yd(n5) ;
      Real_t yv6 = domain.yd(n6) ;
      Real_t yv7 = domain.yd(n7) ;

      Real_t zv0 = domain.zd(n0) ;
      Real_t zv1 = domain.zd(n1) ;
      Real_t zv2 = domain.zd(n2) ;
      Real_t zv3 = domain.zd(n3) ;
      Real_t zv4 = domain.zd(n4) ;
      Real_t zv5 = domain.zd(n5) ;
      Real_t zv6 = domain.zd(n6) ;
      Real_t zv7 = domain.zd(n7) ;

      Real_t vol = domain.volo(i)*domain.vnew(i) ;
      Real_t norm = Real_t(1.0) / ( vol + ptiny ) ;

      Real_t dxj = Real_t(-0.25)*(SUM4(x0,x1,x5,x4) - SUM4(x3,x2,x6,x7)) ;
      Real_t dyj = Real_t(-0.25)*(SUM4(y0,y1,y5,y4) - SUM4(y3,y2,y6,y7)) ;
      Real_t dzj = Real_t(-0.25)*(SUM4(z0,z1,z5,z4) - SUM4(z3,z2,z6,z7)) ;

      Real_t dxi = Real_t( 0.25)*(SUM4(x1,x2,x6,x5) - SUM4(x0,x3,x7,x4)) ;
      Real_t dyi = Real_t( 0.25)*(SUM4(y1,y2,y6,y5) - SUM4(y0,y3,y7,y4)) ;
      Real_t dzi = Real_t( 0.25)*(SUM4(z1,z2,z6,z5) - SUM4(z0,z3,z7,z4)) ;

      Real_t dxk = Real_t( 0.25)*(SUM4(x4,x5,x6,x7) - SUM4(x0,x1,x2,x3)) ;
      Real_t dyk = Real_t( 0.25)*(SUM4(y4,y5,y6,y7) - SUM4(y0,y1,y2,y3)) ;
      Real_t dzk = Real_t( 0.25)*(SUM4(z4,z5,z6,z7) - SUM4(z0,z1,z2,z3)) ;

      /* find delvk and delxk ( i cross j ) */

      ax = dyi*dzj - dzi*dyj ;
      ay = dzi*dxj - dxi*dzj ;
      az = dxi*dyj - dyi*dxj ;

      domain.delx_zeta(i) = vol / SQRT(ax*ax + ay*ay + az*az + ptiny) ;

      ax *= norm ;
      ay *= norm ;
      az *= norm ;

      dxv = Real_t(0.25)*(SUM4(xv4,xv5,xv6,xv7) - SUM4(xv0,xv1,xv2,xv3)) ;
      dyv = Real_t(0.25)*(SUM4(yv4,yv5,yv6,yv7) - SUM4(yv0,yv1,yv2,yv3)) ;
      dzv = Real_t(0.25)*(SUM4(zv4,zv5,zv6,zv7) - SUM4(zv0,zv1,zv2,zv3)) ;

      domain.delv_zeta(i) = ax*dxv + ay*dyv + az*dzv ;

      /* find delxi and delvi ( j cross k ) */

      ax = dyj*dzk - dzj*dyk ;
      ay = dzj*dxk - dxj*dzk ;
      az = dxj*dyk - dyj*dxk ;

      domain.delx_xi(i) = vol / SQRT(ax*ax + ay*ay + az*az + ptiny) ;

      ax *= norm ;
      ay *= norm ;
      az *= norm ;

      dxv = Real_t(0.25)*(SUM4(xv1,xv2,xv6,xv5) - SUM4(xv0,xv3,xv7,xv4)) ;
      dyv = Real_t(0.25)*(SUM4(yv1,yv2,yv6,yv5) - SUM4(yv0,yv3,yv7,yv4)) ;
      dzv = Real_t(0.25)*(SUM4(zv1,zv2,zv6,zv5) - SUM4(zv0,zv3,zv7,zv4)) ;

      domain.delv_xi(i) = ax*dxv + ay*dyv + az*dzv ;

      /* find delxj and delvj ( k cross i ) */

      ax = dyk*dzi - dzk*dyi ;
      ay = dzk*dxi - dxk*dzi ;
      az = dxk*dyi - dyk*dxi ;

      domain.delx_eta(i) = vol / SQRT(ax*ax + ay*ay + az*az + ptiny) ;

      ax *= norm ;
      ay *= norm ;
      az *= norm ;

      dxv = Real_t(-0.25)*(SUM4(xv0,xv1,xv5,xv4) - SUM4(xv3,xv2,xv6,xv7)) ;
      dyv = Real_t(-0.25)*(SUM4(yv0,yv1,yv5,yv4) - SUM4(yv3,yv2,yv6,yv7)) ;
      dzv = Real_t(-0.25)*(SUM4(zv0,zv1,zv5,zv4) - SUM4(zv3,zv2,zv6,zv7)) ;

      domain.delv_eta(i) = ax*dxv + ay*dyv + az*dzv ;
   }
#undef SUM4
}

void Lulesh::CalcMonotonicQRegionForElems(// parameters
      Real_t qlc_monoq,
      Real_t qqc_monoq,
      Real_t monoq_limiter_mult,
      Real_t monoq_max_slope,
      Real_t ptiny,

      // the elementset length
      Index_t elength )
{
   for ( Index_t ielem = 0 ; ielem < elength; ++ielem ) {
      Real_t qlin, qquad ;
      Real_t phixi, phieta, phizeta ;
      Index_t i = domain.matElemlist(ielem);
      Int_t bcMask = domain.elemBC(i) ;
      Real_t delvm, delvp ;

      /*  phixi     */
      Real_t norm = Real_t(1.) / ( domain.delv_xi(i) + ptiny ) ;

      switch (bcMask & XI_M) {
         case XI_M_COMM: /* needs comm data */
         case 0:         delvm = domain.delv_xi(domain.lxim(i)) ; break ;
         case XI_M_SYMM: delvm = domain.delv_xi(i) ;              break ;
         case XI_M_FREE: delvm = Real_t(0.0) ;                    break ;
         default:        /* ERROR */ ;                            break ;
      }
      switch (bcMask & XI_P) {
         case XI_P_COMM: /* needs comm data */
         case 0:         delvp = domain.delv_xi(domain.lxip(i)) ; break ;
         case XI_P_SYMM: delvp = domain.delv_xi(i) ;              break ;
         case XI_P_FREE: delvp = Real_t(0.0) ;                    break ;
         default:        /* ERROR */ ;                            break ;
      }

      delvm = delvm * norm ;
      delvp = delvp * norm ;

      phixi = Real_t(.5) * ( delvm + delvp ) ;

      delvm *= monoq_limiter_mult ;
      delvp *= monoq_limiter_mult ;

      if ( delvm < phixi ) phixi = delvm ;
      if ( delvp < phixi ) phixi = delvp ;
      if ( phixi < Real_t(0.)) phixi = Real_t(0.) ;
      if ( phixi > monoq_max_slope) phixi = monoq_max_slope;


      /*  phieta     */
      norm = Real_t(1.) / ( domain.delv_eta(i) + ptiny ) ;

      switch (bcMask & ETA_M) {
         case 0:          delvm = domain.delv_eta(domain.letam(i)) ; break ;
         case ETA_M_SYMM: delvm = domain.delv_eta(i) ;               break ;
         case ETA_M_FREE: delvm = Real_t(0.0) ;                      break ;
         default:         /* ERROR */ ;                              break ;
      }
      switch (bcMask & ETA_P) {
         case 0:          delvp = domain.delv_eta(domain.letap(i)) ; break ;
         case ETA_P_SYMM: delvp = domain.delv_eta(i) ;               break ;
         case ETA_P_FREE: delvp = Real_t(0.0) ;                      break ;
         default:         /* ERROR */ ;                              break ;
      }

      delvm = delvm * norm ;
      delvp = delvp * norm ;

      phieta = Real_t(.5) * ( delvm + delvp ) ;

      delvm *= monoq_limiter_mult ;
      delvp *= monoq_limiter_mult ;

      if ( delvm  < phieta ) phieta = delvm ;
      if ( delvp  < phieta ) phieta = delvp ;
      if ( phieta < Real_t(0.)) phieta = Real_t(0.) ;
      if ( phieta > monoq_max_slope)  phieta = monoq_max_slope;

      /*  phizeta     */
      norm = Real_t(1.) / ( domain.delv_zeta(i) + ptiny ) ;

      switch (bcMask & ZETA_M) {
         case 0:           delvm = domain.delv_zeta(domain.lzetam(i)) ; break ;
         case ZETA_M_SYMM: delvm = domain.delv_zeta(i) ;                break ;
         case ZETA_M_FREE: delvm = Real_t(0.0) ;                        break ;
         default:          /* ERROR */ ;                                break ;
      }
      switch (bcMask & ZETA_P) {
         case 0:           delvp = domain.delv_zeta(domain.lzetap(i)) ; break ;
         case ZETA_P_SYMM: delvp = domain.delv_zeta(i) ;                break ;
         case ZETA_P_FREE: delvp = Real_t(0.0) ;                        break ;
         default:          /* ERROR */ ;                                break ;
      }

      delvm = delvm * norm ;
      delvp = delvp * norm ;

      phizeta = Real_t(.5) * ( delvm + delvp ) ;

      delvm *= monoq_limiter_mult ;
      delvp *= monoq_limiter_mult ;

      if ( delvm   < phizeta ) phizeta = delvm ;
      if ( delvp   < phizeta ) phizeta = delvp ;
      if ( phizeta < Real_t(0.)) phizeta = Real_t(0.);
      if ( phizeta > monoq_max_slope  ) phizeta = monoq_max_slope;

      /* Remove length scale */

      if ( domain.vdov(i) > Real_t(0.) )  {
         qlin  = Real_t(0.) ;
         qquad = Real_t(0.) ;
      }
      else {
         Real_t delvxxi   = domain.delv_xi(i)   * domain.delx_xi(i)   ;
         Real_t delvxeta  = domain.delv_eta(i)  * domain.delx_eta(i)  ;
         Real_t delvxzeta = domain.delv_zeta(i) * domain.delx_zeta(i) ;

         if ( delvxxi   > Real_t(0.) ) delvxxi   = Real_t(0.) ;
         if ( delvxeta  > Real_t(0.) ) delvxeta  = Real_t(0.) ;
         if ( delvxzeta > Real_t(0.) ) delvxzeta = Real_t(0.) ;

         Real_t rho = domain.elemMass(i) / (domain.volo(i) * domain.vnew(i)) ;

         qlin = -qlc_monoq * rho *
            (  delvxxi   * (Real_t(1.) - phixi) +
               delvxeta  * (Real_t(1.) - phieta) +
               delvxzeta * (Real_t(1.) - phizeta)  ) ;

         qquad = qqc_monoq * rho *
            (  delvxxi*delvxxi     * (Real_t(1.) - phixi*phixi) +
               delvxeta*delvxeta   * (Real_t(1.) - phieta*phieta) +
               delvxzeta*delvxzeta * (Real_t(1.) - phizeta*phizeta)  ) ;
      }

      domain.qq(i) = qquad ;
      domain.ql(i) = qlin  ;
   }
}

void Lulesh::CalcMonotonicQForElems()
{  
   //
   // initialize parameters
   // 
   const Real_t ptiny        = Real_t(1.e-36) ;
   Real_t monoq_max_slope    = domain.monoq_max_slope() ;
   Real_t monoq_limiter_mult = domain.monoq_limiter_mult() ;

   //
   // calculate the monotonic q for pure regions
   //
   Index_t elength = domain.numElem() ;
   if (elength > 0) {
      Real_t qlc_monoq = domain.qlc_monoq();
      Real_t qqc_monoq = domain.qqc_monoq();
      CalcMonotonicQRegionForElems(// parameters
            qlc_monoq,
            qqc_monoq,
            monoq_limiter_mult,
            monoq_max_slope,
            ptiny,

            // the elemset length
            elength );
   }
}

void Lulesh::CalcQForElems()
{
   Real_t qstop = domain.qstop() ;
   Index_t numElem = domain.numElem() ;

   //
   // MONOTONIC Q option
   //

#if defined(COEVP_MPI)
   // Real_t *fieldData[3] ;
   Real_t *fieldData[1] ;

   CommRecv(&domain, MSG_MONOQ, 1 /* 3 */, domain.commElems()) ;
#endif

   /* Calculate velocity gradients */
   CalcMonotonicQGradientsForElems() ;

#if defined(COEVP_MPI)

   /* Transfer veloctiy gradients in the first order elements */

   fieldData[0] = &domain.delv_xi(0) ;
   // fieldData[1] = &domain.delv_eta(0) ;
   // fieldData[2] = &domain.delv_zeta(0) ;

   showMeMonoQ = 0 ;
   CommSend(&domain, MSG_MONOQ, 1 /* 3 */, fieldData,
         domain.planeElemIds, domain.commElems(), domain.sliceHeight() - 1) ;
   CommMonoQ(&domain, domain.planeElemIds, domain.commElems(), domain.sliceHeight() - 1) ;
   showMeMonoQ = 0 ;

#endif

}

void Lulesh::CalcQForElems2()
{

   Real_t qstop = domain.qstop() ;
   Index_t numElem = domain.numElem() ;

   CalcMonotonicQForElems() ;

   /* Don't allow excessive artificial viscosity */
   if (numElem != 0) {
      Index_t idx = -1; 
      for (Index_t i=0; i<numElem; ++i) {
         if ( domain.q(i) > qstop ) {
            idx = i ;
            break ;
         }
      }

      if(idx >= 0) {
         cout << "At element " << idx << ", q = " << domain.q(idx) <<
            ", qstop = " << qstop << endl;
#if defined(COEVP_MPI)
         MPI_Abort(MPI_COMM_WORLD, QStopError) ;
#else
         exit(QStopError) ;
#endif
      }
   }
}

void Lulesh::CalcPressureForElems(Real_t* p_new, Real_t* bvc,
      Real_t* pbvc, Real_t* e_old,
      Real_t* compression, Real_t *vnewc,
      Real_t pmin,
      Real_t p_cut, Real_t eosvmax,
      Index_t length)
{
   Real_t c1s = Real_t(2.0)/Real_t(3.0) ;
   for (Index_t i = 0; i < length ; ++i) {
      bvc[i] = c1s * (compression[i] + Real_t(1.));
      pbvc[i] = c1s;
   }

   for (Index_t i = 0 ; i < length ; ++i){
      p_new[i] = domain.cm(i)->pressure(compression[i], e_old[i]) ;
   }
}

void Lulesh::CalcEnergyForElems(Real_t* p_new, Real_t* e_new, Real_t* q_new,
      Real_t* bvc, Real_t* pbvc,
      Real_t* p_old, Real_t* e_old, Real_t* q_old,
      Real_t* compression, Real_t* compHalfStep,
      Real_t* vnewc, Real_t* work, Real_t* delvc, Real_t pmin,
      Real_t p_cut, Real_t  e_cut, Real_t q_cut, Real_t emin,
      Real_t* qq, Real_t* ql,
      Real_t rho0,
      Real_t eosvmax,
      Index_t length)
{
   const Real_t sixth = Real_t(1.0) / Real_t(6.0) ;
   Real_t *pHalfStep = Allocate<Real_t>(length) ;

   for (Index_t i = 0 ; i < length ; ++i) {
      e_new[i] = e_old[i] - Real_t(0.5) * delvc[i] * (p_old[i] + q_old[i])
         + Real_t(0.5) * work[i];

      if (e_new[i]  < emin ) {
         e_new[i] = emin ;
      }
   }

   CalcPressureForElems(pHalfStep, bvc, pbvc, e_new, compHalfStep, vnewc,
         pmin, p_cut, eosvmax, length);

   for (Index_t i = 0 ; i < length ; ++i) {
      Real_t vhalf = Real_t(1.) / (Real_t(1.) + compHalfStep[i]) ;

      if ( delvc[i] > Real_t(0.) ) {
         q_new[i] /* = qq[i] = ql[i] */ = Real_t(0.) ;
      }
      else {
         Real_t ssc = domain.cm(i)->soundSpeedSquared(rho0, vhalf, e_new[i]);

         if ( ssc <= Real_t(0.) ) {
            ssc =Real_t(.333333e-36) ;
         } else {
            ssc = SQRT(ssc) ;
         }

         q_new[i] = (ssc*ql[i] + qq[i]) ;
      }

      e_new[i] = e_new[i] + Real_t(0.5) * delvc[i]
         * (  Real_t(3.0)*(p_old[i]     + q_old[i])
               - Real_t(4.0)*(pHalfStep[i] + q_new[i])) ;
   }

   for (Index_t i = 0 ; i < length ; ++i) {

      e_new[i] += Real_t(0.5) * work[i];

      if (FABS(e_new[i]) < e_cut) {
         e_new[i] = Real_t(0.)  ;
      }
      if (     e_new[i]  < emin ) {
         e_new[i] = emin ;
      }
   }

   CalcPressureForElems(p_new, bvc, pbvc, e_new, compression, vnewc,
         pmin, p_cut, eosvmax, length);

   for (Index_t i = 0 ; i < length ; ++i){
      Real_t q_tilde ;

      if (delvc[i] > Real_t(0.)) {
         q_tilde = Real_t(0.) ;
      }
      else {

         Real_t ssc = domain.cm(i)->soundSpeedSquared(rho0, vnewc[i], e_new[i]);

         if ( ssc <= Real_t(0.) ) {
            ssc = Real_t(.333333e-36) ;
         } else {
            ssc = SQRT(ssc) ;
         }

         q_tilde = (ssc*ql[i] + qq[i]) ;
      }

      e_new[i] = e_new[i] - (  Real_t(7.0)*(p_old[i]     + q_old[i])
            - Real_t(8.0)*(pHalfStep[i] + q_new[i])
            + (p_new[i] + q_tilde)) * delvc[i]*sixth ;

      if (FABS(e_new[i]) < e_cut) {
         e_new[i] = Real_t(0.)  ;
      }
      if (     e_new[i]  < emin ) {
         e_new[i] = emin ;
      }
   }

   CalcPressureForElems(p_new, bvc, pbvc, e_new, compression, vnewc,
         pmin, p_cut, eosvmax, length);

   for (Index_t i = 0 ; i < length ; ++i){

      if ( delvc[i] <= Real_t(0.) ) {

         Real_t ssc = domain.cm(i)->soundSpeedSquared(rho0, vnewc[i], e_new[i]);

         if ( ssc <= Real_t(0.) ) {
            ssc = Real_t(.333333e-36) ;
         } else {
            ssc = SQRT(ssc) ;
         }

         q_new[i] = (ssc*ql[i] + qq[i]) ;

         if (FABS(q_new[i]) < q_cut) q_new[i] = Real_t(0.) ;
      }
   }

   Release(&pHalfStep) ;

   return ;
}

void Lulesh::CalcSoundSpeedForElems(Real_t *vnewc, Real_t rho0, Real_t *enewc,
      Real_t *pnewc, Real_t *pbvc,
      Real_t *bvc, Real_t ss4o3, Index_t nz)
{
   for (Index_t i = 0; i < nz ; ++i) {
      Index_t iz = domain.matElemlist(i);

      Real_t ssTmp = domain.cm(i)->soundSpeedSquared(rho0, vnewc[i], enewc[i]);

      if (ssTmp <= Real_t(1.111111e-36)) {
         ssTmp = Real_t(1.111111e-36);
      }
      domain.ss(iz) = SQRT(ssTmp);
   }
}


void Lulesh::CalcWorkForElems(Real_t *vc, Real_t *work, Index_t length)
{
   Real_t dt = domain.deltatime();

   for (Index_t i = 0; i < length ; ++i) {

#if 1
      work[i] = vc[i] * (domain.sx(i)*domain.dxx(i) +
            domain.sy(i)*domain.dyy(i) +
            (-domain.sx(i)-domain.sy(i))*domain.dzz(i)
            + 2. * (domain.txy(i)*domain.dxy(i) +
               domain.txz(i)*domain.dxz(i) +
               domain.tyz(i)*domain.dyz(i)) );
#else
      work[i] = Real_t(0.);
#endif
      work[i] *= dt;
   }
}

void Lulesh::EvalEOSForElems(Real_t *vnewc, Index_t length)
{
   Real_t  e_cut = domain.e_cut();
   Real_t  p_cut = domain.p_cut();
   Real_t  ss4o3 = domain.ss4o3();
   Real_t  q_cut = domain.q_cut();

   Real_t eosvmax = domain.eosvmax() ;
   Real_t eosvmin = domain.eosvmin() ;
   Real_t pmin    = domain.pmin() ;
   Real_t emin    = domain.emin() ;
   Real_t rho0    = domain.refdens() ;

   Real_t *e_old = Allocate<Real_t>(length) ;
   Real_t *delvc = Allocate<Real_t>(length) ;
   Real_t *p_old = Allocate<Real_t>(length) ;
   Real_t *q_old = Allocate<Real_t>(length) ;
   Real_t *compression = Allocate<Real_t>(length) ;
   Real_t *compHalfStep = Allocate<Real_t>(length) ;
   Real_t *qq = Allocate<Real_t>(length) ;
   Real_t *ql = Allocate<Real_t>(length) ;
   Real_t *work = Allocate<Real_t>(length) ;
   Real_t *p_new = Allocate<Real_t>(length) ;
   Real_t *e_new = Allocate<Real_t>(length) ;
   Real_t *q_new = Allocate<Real_t>(length) ;
   Real_t *bvc = Allocate<Real_t>(length) ;
   Real_t *pbvc = Allocate<Real_t>(length) ;

   /* compress data, minimal set */
   for (Index_t i=0; i<length; ++i) {
      Index_t zidx = domain.matElemlist(i) ;
      e_old[i] = domain.e(zidx) ;
   }

   for (Index_t i=0; i<length; ++i) {
      Index_t zidx = domain.matElemlist(i) ;
      delvc[i] = domain.delv(zidx) ;
   }

   for (Index_t i=0; i<length; ++i) {
      Index_t zidx = domain.matElemlist(i) ;
      p_old[i] = domain.p(zidx) ;
   }

   for (Index_t i=0; i<length; ++i) {
      Index_t zidx = domain.matElemlist(i) ;
      q_old[i] = domain.q(zidx) ;
   }

   for (Index_t i = 0; i < length ; ++i) {
      Real_t vchalf ;
      compression[i] = Real_t(1.) / vnewc[i] - Real_t(1.);
      vchalf = vnewc[i] - delvc[i] * Real_t(.5);
      compHalfStep[i] = Real_t(1.) / vchalf - Real_t(1.);
   }

   /* Check for v > eosvmax or v < eosvmin */
   if ( eosvmin != Real_t(0.) ) {
      for(Index_t i=0 ; i<length ; ++i) {
         if (vnewc[i] <= eosvmin) { /* impossible due to calling func? */
            compHalfStep[i] = compression[i] ;
         }
      }
   }
   if ( eosvmax != Real_t(0.) ) {
      for(Index_t i=0 ; i<length ; ++i) {
         if (vnewc[i] >= eosvmax) { /* impossible due to calling func? */
            p_old[i]        = Real_t(0.) ;
            compression[i]  = Real_t(0.) ;
            compHalfStep[i] = Real_t(0.) ;
         }
      }
   }

   for (Index_t i = 0 ; i < length ; ++i) {
      Index_t zidx = domain.matElemlist(i) ;
      qq[i] = domain.qq(zidx) ;
      ql[i] = domain.ql(zidx) ;
   }

   CalcWorkForElems(vnewc, work, length);

   CalcEnergyForElems(p_new, e_new, q_new, bvc, pbvc,
         p_old, e_old,  q_old, compression, compHalfStep,
         vnewc, work,  delvc, pmin,
         p_cut, e_cut, q_cut, emin,
         qq, ql, rho0, eosvmax, length);


   for (Index_t i=0; i<length; ++i) {
      Index_t zidx = domain.matElemlist(i) ;
      domain.p(zidx) = p_new[i] ;
   }

   for (Index_t i=0; i<length; ++i) {
      Index_t zidx = domain.matElemlist(i) ;
      domain.e(zidx) = e_new[i] ;
   }

   for (Index_t i=0; i<length; ++i) {
      Index_t zidx = domain.matElemlist(i) ;
      domain.q(zidx) = q_new[i] ;
   }

   CalcSoundSpeedForElems(vnewc, rho0, e_new, p_new,
         pbvc, bvc, ss4o3, length) ;

   Release(&pbvc) ;
   Release(&bvc) ;
   Release(&q_new) ;
   Release(&e_new) ;
   Release(&p_new) ;
   Release(&work) ;
   Release(&ql) ;
   Release(&qq) ;
   Release(&compHalfStep) ;
   Release(&compression) ;
   Release(&q_old) ;
   Release(&p_old) ;
   Release(&delvc) ;
   Release(&e_old) ;
}

void Lulesh::ApplyMaterialPropertiesForElems()
{
   Index_t length = domain.numElem() ;

   if (length != 0) {
      /* Expose all of the variables needed for material evaluation */
      Real_t eosvmin = domain.eosvmin() ;
      Real_t eosvmax = domain.eosvmax() ;
      Real_t *vnewc = Allocate<Real_t>(length) ;

      for (Index_t i=0 ; i<length ; ++i) {
         Index_t zn = domain.matElemlist(i) ;
         vnewc[i] = domain.vnew(zn) ;
      }

      if (eosvmin != Real_t(0.)) {
         for(Index_t i=0 ; i<length ; ++i) {
            if (vnewc[i] < eosvmin)
               vnewc[i] = eosvmin ;
         }
      }

      if (eosvmax != Real_t(0.)) {
         for(Index_t i=0 ; i<length ; ++i) {
            if (vnewc[i] > eosvmax)
               vnewc[i] = eosvmax ;
         }
      }

      for (Index_t i=0; i<length; ++i) {
         Index_t zn = domain.matElemlist(i) ;
         Real_t vc = domain.v(zn) ;
         if (eosvmin != Real_t(0.)) {
            if (vc < eosvmin)
               vc = eosvmin ;
         }
         if (eosvmax != Real_t(0.)) {
            if (vc > eosvmax)
               vc = eosvmax ;
         }
         if (vc <= 0.) {
#if defined(COEVP_MPI)
            MPI_Abort(MPI_COMM_WORLD, VolumeError) ;
#else
            exit(VolumeError) ;
#endif
         }
      }

      EvalEOSForElems(vnewc, length);

      Release(&vnewc) ;

   }
}

void Lulesh::UpdateVolumesForElems()
{
   Index_t numElem = domain.numElem();
   if (numElem != 0) {
      Real_t v_cut = domain.v_cut();

      for(Index_t i=0 ; i<numElem ; ++i) {
         Real_t tmpV ;
         tmpV = domain.vnew(i) ;

         if ( FABS(tmpV - Real_t(1.0)) < v_cut )
            tmpV = Real_t(1.0) ;
         domain.v(i) = tmpV ;
      }
   }

   return ;
}

void Lulesh::LagrangeElements()
{
   const Real_t deltatime = domain.deltatime() ;

   CalcLagrangeElements(deltatime) ;

   /* Calculate Q.  (Monotonic q option requires communication) */
   CalcQForElems() ;

}

void Lulesh::LagrangeElements2(){
   CalcQForElems2();

   ApplyMaterialPropertiesForElems() ;

   UpdateVolumesForElems() ;
}

void Lulesh::CalcCourantConstraintForElems()
{
   Real_t dtcourant = Real_t(1.0e+20) ;
   Index_t   courant_elem = -1 ;
   Real_t      qqc = domain.qqc() ;
   Index_t length = domain.numElem() ;

   Real_t  qqc2 = Real_t(64.0) * qqc * qqc ;

   for (Index_t i = 0 ; i < length ; ++i) {
      Index_t indx = domain.matElemlist(i) ;

      Real_t dtf = domain.ss(indx) * domain.ss(indx) ;

      if ( domain.vdov(indx) < Real_t(0.) ) {

         dtf = dtf
            + qqc2 * domain.arealg(indx) * domain.arealg(indx)
            * domain.vdov(indx) * domain.vdov(indx) ;
      }

      dtf = SQRT(dtf) ;

      dtf = domain.arealg(indx) / dtf ;

      /* determine minimum timestep with its corresponding elem */
      if (domain.vdov(indx) != Real_t(0.)) {
         if ( dtf < dtcourant ) {
            dtcourant = dtf ;
            courant_elem = indx ;
         }
      }
   }

   /* Don't try to register a time constraint if none of the elements
    * were active */
   if (courant_elem != -1) {
      domain.dtcourant() = dtcourant ;
   }

   return ;
}

void Lulesh::CalcHydroConstraintForElems()
{
   Real_t dthydro = Real_t(1.0e+20) ;
   Index_t hydro_elem = -1 ;
   Real_t dvovmax = domain.dvovmax() ;
   Index_t length = domain.numElem() ;

   for (Index_t i = 0 ; i < length ; ++i) {
      Index_t indx = domain.matElemlist(i) ;

      if (domain.vdov(indx) != Real_t(0.)) {
         Real_t dtdvov = dvovmax / (FABS(domain.vdov(indx))+Real_t(1.e-20)) ;
         if ( dthydro > dtdvov ) {
            dthydro = dtdvov ;
            hydro_elem = indx ;
         }
      }
   }

   if (hydro_elem != -1) {
      domain.dthydro() = dthydro ;
   }

   return ;
}

void Lulesh::CalcTimeConstraintsForElems() {
   /* evaluate time constraint */
   CalcCourantConstraintForElems() ;

   /* check hydro constraint */
   CalcHydroConstraintForElems() ;
}

void Lulesh::LagrangeLeapFrog()
{
#if defined(COEVP_MPI) && defined(SEDOV_SYNC_POS_VEL_LATE)
   Real_t *fieldData[6] ;
#endif

   /* calculate nodal forces, accelerations, velocities, positions, with
    * applied boundary conditions and slide surface considerations */
   LagrangeNodal();

   /* calculate element quantities (i.e. velocity gradient & q), and update
    * material states */
   LagrangeElements();
   LagrangeElements2();

#if defined(COEVP_MPI) && defined(SEDOV_SYNC_POS_VEL_LATE)
   CommRecv(&domain, MSG_SYNC_POS_VEL, 6, domain.commNodes(), false) ;

   /* !!! May need more time between recv and send !!! */

   fieldData[0] = &domain.x(0) ;
   fieldData[1] = &domain.y(0) ;
   fieldData[2] = &domain.z(0) ;
   fieldData[3] = &domain.xd(0) ;
   fieldData[4] = &domain.yd(0) ;
   fieldData[5] = &domain.zd(0) ;

   CommSend(&domain, MSG_SYNC_POS_VEL, 6, fieldData,
         domain.planeNodeIds, domain.commNodes(), domain.sliceHeight(),
         false) ;
#endif

   CalcTimeConstraintsForElems();

#if defined(COEVP_MPI) && defined(SEDOV_SYNC_POS_VEL_LATE)
   CommSyncPosVel(&domain,
         domain.planeNodeIds, domain.commNodes(), domain.sliceHeight()) ;
#endif

   // LagrangeRelease() ;  Creation/destruction of temps may be important to capture 
}

void Lulesh::FinalTime()
{
	if(timer)
	{
		//Do one final probe. We will be off by a small amount but it should be fairly insignificant for any meaningful run and it is higher than our actual time anyway
		if(!time_output)
		{
			std::list<std::chrono::high_resolution_clock::time_point>::iterator it;
			for (int i=1;i<domain.cycle();i++)
			{
				int t_count = i;
		        int scale=0;
    		    if(timer != 1)
      	  		{
	        	    while ( t_count /= timer)
    	        	scale++;
	    	        scale = pow(timer, scale);
	    	    }
		        else
    		    {
	        	    scale = 1;
		        }


				if(i == scale)
		        {
            
	        	    if(scale == 1)
		            {
						it=timings.begin();
	    	         	timerfile << "Timer Output Frequency is " << scale << std::endl;
	            	}
		            else
		            {   
						it++;	
	    	            timerfile  << "Changing Timer Output Frequency to " << scale << std::endl;
	        	        std::chrono::duration<double> diff = *it - timings.front();
	            	    timerfile  << "0 - " << i << ": " << diff.count() << " s" << std::endl;
		            }
				}
				else
				{
		            if(i % scale == 0)
	    	        {
						it ++;
	            	    std::chrono::duration<double> diff = *it - *std::prev(it,1);
	                	timerfile  << i - scale << " - " << i << ": " << diff.count() << " s" << std::endl;
		            }
				}
			}

		}
		timings.push_back(std::chrono::high_resolution_clock::now());
		std::chrono::duration<double> diff = timings.back() - timings.front();
		timerfile  << "Total Cycles: " << domain.cycle() << " Time: " << diff.count() << " s" << std::endl;
	}
 }

void Lulesh::OutputTiming()
{
   if(timer)
	{
        int t_count = domain.cycle();
		int scale=0;
		if(timer != 1)
		{
	        while ( t_count /= timer)
	           scale++;
			scale = pow(timer, scale);
		}
		else
		{
			scale = 1;
		}


		if(domain.cycle() == scale)
		{
			
			timings.push_back(std::chrono::high_resolution_clock::now());
			if(scale == 1 && time_output)
			{
				timerfile << "Timer Output Frequency is " << scale << std::endl;
			}
			else if(time_output)
			{	
				timerfile  << "Changing Timer Output Frequency to " << scale << std::endl;
				std::chrono::duration<double> diff = timings.back() - timings.front();
				timerfile  << "0 - " << domain.cycle() << ": " << diff.count() << " s" << std::endl;
			}
		}
		else
		{
			if(domain.cycle() % scale == 0)
			{
				timings.push_back(std::chrono::high_resolution_clock::now());
				if(time_output)
				{
					std::chrono::duration<double> diff = timings.back() - *std::prev(timings.end(),2);
					timerfile  << domain.cycle() - scale << " - " << domain.cycle() << ": " << diff.count() << " s" << std::endl;
				}
			}
		}

	}


}

int Lulesh::UpdateStressForElems()
{
   //#define MAX_NONLINEAR_ITER 5
   int max_nonlinear_iters = 0;
   int numElem = domain.numElem() ;


#ifdef _OPENMP
#pragma omp parallel
#endif
   {
      int max_local_newton_iters = 0;

#ifdef _OPENMP
#pragma omp for
#endif
      for (Index_t k=0; k<numElem; ++k) {

#ifdef FSTRACE
         cout << "Processing FS element " << k << endl;
#endif

#if defined(PROTOBUF)
         struct WrapReturn *wrap_ret = wrap_advance(domain, k);
         ConstitutiveData cm_data = *(wrap_ret->cm_data);
         delete wrap_ret;
#else
         ConstitutiveData cm_data = domain.cm(k)->advance(domain.deltatime(),
                                                          domain.cm_vel_grad(k),
                                                          domain.cm_vol_chng(k),
                                                          domain.cm_state(k));
#endif
         int num_iters = cm_data.num_Newton_iters;
         if (num_iters > max_local_newton_iters) max_local_newton_iters = num_iters;

         const Tensor2Sym& sigma_prime = cm_data.sigma_prime;

         Real_t sx  = domain.sx(k) = sigma_prime(1,1);
         Real_t sy  = domain.sy(k) = sigma_prime(2,2);
         Real_t sz  = - sx - sy;
         Real_t txy = domain.txy(k) = sigma_prime(2,1);
         Real_t txz = domain.txz(k) = sigma_prime(3,1);
         Real_t tyz = domain.tyz(k) = sigma_prime(3,2);

         domain.mises(k) = SQRT( Real_t(0.5) * ( (sy - sz)*(sy - sz) + (sz - sx)*(sz - sx) + (sx - sy)*(sx - sy) )
                               + Real_t(3.0) * ( txy*txy + txz*txz + tyz*tyz) );
		

      }



#ifdef _OPENMP
#pragma omp critical
#endif
      {
         if (max_local_newton_iters > max_nonlinear_iters) {
            max_nonlinear_iters = max_local_newton_iters;
         }
      }
   }

#if defined(COEVP_MPI)
   {
      int g_max_nonlinear_iters ;

      MPI_Allreduce(&max_nonlinear_iters, &g_max_nonlinear_iters, 1,
            MPI_INT, MPI_MAX, MPI_COMM_WORLD) ;
      max_nonlinear_iters = g_max_nonlinear_iters ;
   }
#endif

   return max_nonlinear_iters;
}

void Lulesh::UpdateStressForElems2(int max_nonlinear_iters)
{
   // The maximum number of Newton iterations required is an indicaton of
   // fast time scales in the fine-scale model.  If the number of iterations
   // becomes large, we need to reduce the timestep.  It it becomes small,
   // we try to increase the time step.
   if (max_nonlinear_iters > MAX_NONLINEAR_ITER) {
      finescale_dt_modifier *= Real_t(0.95);
   }
   else if(max_nonlinear_iters < 0.5 * MAX_NONLINEAR_ITER) {
      finescale_dt_modifier *= Real_t(1.05);
      if (finescale_dt_modifier > 1.) finescale_dt_modifier = 1.;
   }

#if 0
   MPI_Barrier(MPI_COMM_WORLD) ;

#if defined(PRINT_PERFORMANCE_DIAGNOSTICS) && defined(LULESH_SHOW_PROGRESS)
   cout << "   finescale_dt_modifier = " << finescale_dt_modifier << endl;
   cout << "   Max nonlinear iterations = " << max_nonlinear_iters << endl;
   cout.flush() ;
#endif

   MPI_Barrier(MPI_COMM_WORLD) ;
#endif
}


void Lulesh::Initialize(int myRank, int numRanks, int edgeDim, int heightDim, double domainStopTime, int simStopCycle, int timerSamplingRate)
{
	this->time_output = 0;
	if(myRank == 0)
	{
		if(timerSamplingRate < 0)
		{
			this->time_output = 1;
			this->timer = -timerSamplingRate;
		}
		else if(timerSamplingRate == 0)
		{
			this->timer = 1;
		}
		else
		{
			this->timer = timerSamplingRate;
		}

//		this->timer = timerSamplingRate;
		if(this->timer != 0)
		{
			this->timerfile.open("timer.file");
			if(this->timerfile.is_open())
			{
				///TODO: Figure out implementation neutral way to write configuration... probably using domain
				/*
				for(int i=0;i<argc;i++)
				{
					this->timerfile << argv[i] << " ";
				}
				
				this->timerfile << std::endl;
				*/
			}
			else
			{
				std::cout << "Could not open timer.file" << std::endl;
			}
		}
	}
	else
	{
		this->timer = 0;
	}

   Index_t edgeElems = edgeDim ;
   Index_t gheightElems = heightDim ;
// Index_t gheightElems = 8 ;
// Index_t edgeElems = 4 ;
   Index_t edgeNodes = edgeElems+1 ;

   Index_t xBegin, xEnd ;
   // Real_t ds = Real_t(1.125)/Real_t(edgeElems) ; /* may accumulate roundoff */
   Real_t tx, ty, tz ;
   Index_t nidx, zidx ;
   Index_t domElems ;
   Index_t planeNodes ;
   Index_t planeElems ;

#if defined(COEVP_MPI)||defined(__CHARMC__)
   Index_t chunkSize ;
   Index_t remainder ;

   if (sizeof(Real_t) != 4 && sizeof(Real_t) != 8) {
      printf("MPI operations only support float and double right now...\n");
#if defined(COEVP_MPI)
      MPI_Abort(MPI_COMM_WORLD, -1) ;
#else
      exit(-1);
#endif
   }

#if 0
   if (MAX_FIELDS_PER_MPI_COMM > CACHE_COHERENCE_PAD_REAL) {
      printf("corner element comm buffers too small.  Fix code.\n") ;
      MPI_Abort(MPI_COMM_WORLD, -1) ;
   }

   if (numRanks > gheightElems) {
      printf("error -- must have at least one plane per MPI rank\n") ;
      MPI_Abort(MPI_COMM_WORLD, -1) ;
   }

#endif

   domain.sliceLoc() = myRank ;
   domain.numSlices() = numRanks ;

   chunkSize = gheightElems / numRanks ;
   remainder = gheightElems % numRanks ;
   if (myRank < remainder) {
      xBegin = (chunkSize+1)*myRank ;
      xEnd = xBegin + (chunkSize+1) ;
   }
   else {
      xBegin = (chunkSize+1)*remainder + (myRank - remainder)*chunkSize ;
      xEnd = xBegin + chunkSize ;
   }
   // domain->sizeX = xEnd - xBegin ;

   int heightElems = xEnd - xBegin ;

#else

   domain.sliceLoc() = 0 ;
   domain.numSlices() = 1 ;

   int heightElems = gheightElems ;
   xBegin = 0 ;
   xEnd = heightElems ;

#endif

   domain.sliceHeight() = heightElems ;

   int heightNodes = heightElems+1 ;

   /* get run options to measure various metrics */

   /* ... */

   /**************************************/
   /*   Initialize Taylor cylinder mesh  */
   /**************************************/

   double domain_length[3];
   //   domain_length[0] = Real_t(3.81e-2);
   domain_length[0] = Real_t(3.81e-2 / 2.);
   domain_length[1] = domain_length[2] = Real_t(7.62e-3);

   /* construct a cylinder mesh */

   //   int coreElems = int (0.33333333333333333*edgeElems) ;
   int coreElems = int (0.25*edgeElems) ;
   // int coreElems = int (0.5*edgeElems) ;
   int wingElems = edgeElems - coreElems ;
   int coreNodes = coreElems + 1 ;
   int wingNodes = edgeNodes - coreNodes ;

   /* X axis is axis of symmetry */

   domain.numElem() = coreElems*edgeElems*heightElems +
      coreElems*heightElems*wingElems ;
   domain.numNode() = coreNodes*edgeNodes*heightNodes +
      (coreNodes-1)*heightNodes*wingNodes ;
   domain.numSymmNodesBoundary() = edgeNodes*heightNodes ;
   domain.numSymmNodesImpact()   = edgeNodes*coreNodes +
      wingNodes*(coreNodes-1) ;

   domElems = domain.numElem() ;

#if defined(COEVP_MPI)||defined(__CHARMC__)

   /* allocate a buffer large enough for nodal ghost data */
   Index_t planeMin, planeMax ;
   domain.commElems() = domain.numElem()/heightElems ;
   domain.commNodes() = domain.numNode()/heightNodes ;
   domain.maxPlaneSize() = CACHE_ALIGN_REAL(domain.numNode()/heightNodes) ;

   /* assume communication to 2 neighbors by default */
   planeMin = planeMax = 1 ;
   if (domain.sliceLoc() == 0) {
      planeMin = 0 ;
   }
   if (domain.sliceLoc() == numRanks-1) {
      planeMax = 0 ;
   }
   /* account for face communication */
   Index_t comBufSize =
      (planeMin + planeMax) *
      domain.maxPlaneSize() * MAX_FIELDS_PER_MPI_COMM ;

   if (comBufSize > 0) {
      domain.commDataSend = new Real_t[comBufSize] ;
      domain.commDataRecv = new Real_t[comBufSize] ;
      /* prevent floating point exceptions */
      memset(domain.commDataSend, 0, comBufSize*sizeof(Real_t)) ;
      memset(domain.commDataRecv, 0, comBufSize*sizeof(Real_t)) ;

      if (domain.commElems() > 0) // SMM
         domain.planeElemIds = new Index_t[domain.commElems()] ;
      else
         exit(-1);
      if (domain.commNodes() > 0) // SMM
         domain.planeNodeIds = new Index_t[domain.commNodes()] ;
      else
         exit(-1);
   }
   else {
      domain.commDataSend = 0 ;
      domain.commDataRecv = 0 ;

      domain.planeElemIds = 0 ;
      domain.planeNodeIds = 0 ;
   }

   /* SYMM_Z only needs to  be allocated on proc 0 */

#endif

   /* allocate field memory */

   domain.AllocateElemPersistent(domain.numElem()) ;
#if defined(COEVP_MPI)||defined(__CHARMC__)
   domain.AllocateElemTemporary (domain.numElem(), domain.commElems()) ;
#else
   domain.AllocateElemTemporary (domain.numElem(), 0) ;
#endif

   domain.AllocateNodalPersistent(domain.numNode()) ;

   domain.AllocateNodesets(domain.numSymmNodesBoundary(),
         domain.numSymmNodesImpact()) ;

   /* initialize nodal coordinates */

   /* build logical space block of dim (edgeNodes, edgeNode, coreNodes) */

   /* build core */
   nidx = 0 ;
   planeNodes = 0 ;
   tz  = Real_t(0.) ;
   for (Index_t plane=0; plane<coreNodes; ++plane) {
      ty = Real_t(0.) ;
      for (Index_t row=0; row<coreNodes; ++row) {
#if defined(COEVP_MPI)||defined(__CHARMC__)
         tx = domain_length[0]*Real_t(xBegin)/Real_t(edgeNodes) ;
         if (domain.numSlices() != 1) {
            domain.planeNodeIds[planeNodes++] = nidx ;
         }
         for (Index_t col=xBegin; col<(xEnd+1); ++col)
#else
            tx = Real_t(0.) ;
         for (Index_t col=0; col<heightNodes; ++col)
#endif
         {
            domain.x(nidx) = tx ;
            domain.y(nidx) = ty ;
            domain.z(nidx) = tz ;
            ++nidx ;
            // tx += ds ; /* may accumulate roundoff... */
            tx = domain_length[0]*Real_t(col+1)/Real_t(edgeNodes) ;
         }
         // ty += ds ;  /* may accumulate roundoff... */
         ty = domain_length[1]*Real_t(row+1)/Real_t(edgeNodes) ;
      }
      nidx += heightNodes*wingNodes ;
      // tz += ds ;  /* may accumulate roundoff... */
      tz = domain_length[2]*Real_t(plane+1)/Real_t(edgeNodes) ;
   }

   /* build wing in Y direction */
   nidx = coreNodes*heightNodes ;
   for (Index_t plane=0; plane<coreNodes; ++plane) {
      for (Index_t row=coreNodes; row<edgeNodes; ++row) {

         /* "x" direction in 2d cartesian plane */
         ty = domain_length[1]* /* problem scale factor */

            ( Real_t(coreElems)/Real_t(edgeNodes) + /* core distance */

              (cos(M_PI_4*Real_t(plane)/Real_t(coreNodes-1)) -
               Real_t(coreElems)/Real_t(edgeNodes)) *  /* interpolate */
              Real_t(row+1 - coreNodes)/Real_t(wingNodes) /* wing distance */
            ) ;

         /* "y" direction in 2d cartesian plane */
         tz = domain_length[2]* /* problem scale factor */

            (Real_t(plane)/Real_t(edgeNodes) + /* core distance */

             (sin(M_PI_4*Real_t(plane)/Real_t(coreNodes-1))
              - Real_t(plane)/Real_t(edgeNodes)) *    /* interpolate */
             Real_t(row+1 - coreNodes)/Real_t(wingNodes) /* wing distance */
            );

#if defined(COEVP_MPI)||defined(__CHARMC__)
         tx = domain_length[0]*Real_t(xBegin)/Real_t(edgeNodes) ;
         if (domain.numSlices() != 1) {
            domain.planeNodeIds[planeNodes++] = nidx ;
         }

         for (Index_t col=xBegin; col<(xEnd+1); ++col)
#else
            tx = Real_t(0.) ;

         for (Index_t col=0; col<heightNodes; ++col)
#endif
         {
            domain.x(nidx) = tx ;
            domain.y(nidx) = ty ;
            domain.z(nidx) = tz ;
            ++nidx ;
            // tx += ds ; /* may accumulate roundoff... */
            tx = domain_length[0]*Real_t(col+1)/Real_t(edgeNodes) ;
         }
      }
      nidx += coreNodes*heightNodes ;
      // tz += ds ;  /* may accumulate roundoff... */
   }

   /* build wing in Z direction */
   nidx = coreNodes*edgeNodes*heightNodes ;
   for (Index_t plane=coreNodes; plane<edgeNodes; ++plane) {
      for (Index_t row=0; row<(coreNodes-1); ++row) {

         /* "x" direction in 2d cartesian plane */
         tz = domain_length[2]*  /* problem scale factor */

            ( Real_t(coreElems)/Real_t(edgeNodes) + /* core distance */

              (cos(M_PI_4*Real_t(row)/Real_t(coreNodes-1)) -
               Real_t(coreElems)/Real_t(edgeNodes)) *    /* interpolate */
              Real_t(plane+1 - coreNodes)/Real_t(wingNodes) /* wing dist */
            ) ;

         /* "y" direction in 2d cartesian plane */
         ty = domain_length[1]* /* problem scale factor */

            (Real_t(row)/Real_t(edgeNodes) + /* core distance */

             (sin(M_PI_4*Real_t(row)/Real_t(coreNodes-1))
              - Real_t(row)/Real_t(edgeNodes)) *         /* interpoate */
             Real_t(plane+1-coreNodes)/Real_t(wingNodes)    /* wing dist */
            ) ;

#if defined(COEVP_MPI)||defined(__CHARMC__)
         tx = domain_length[0]*Real_t(xBegin)/Real_t(edgeNodes) ;
         if (domain.numSlices() != 1) {
            domain.planeNodeIds[planeNodes++] = nidx ;
         }

         for (Index_t col=xBegin; col<(xEnd+1); ++col)
#else
            tx = Real_t(0.) ;

         for (Index_t col=0; col<heightNodes; ++col)
#endif
         {
            domain.x(nidx) = tx ;
            domain.y(nidx) = ty ;
            domain.z(nidx) = tz ;
            ++nidx ;
            // tx += ds ; /* may accumulate roundoff... */
            tx = domain_length[0]*Real_t(col+1)/Real_t(edgeNodes) ;
         }
      }
   }

#if defined(COEVP_MPI)||defined(__CHARMC__)
   if (domain.numSlices() != 1) {
      if (planeNodes != domain.commNodes()) {
         printf("error computing comm nodes\n") ;
         exit(-1) ;
      }
   }
#endif

   /* embed hexehedral elements in nodal point lattice */

   nidx = 0 ;
   zidx = 0 ;
   planeElems = 0 ;
   for (Index_t plane=0; plane<coreElems; ++plane) {
      for (Index_t row=0; row<edgeElems; ++row) {
#if defined(COEVP_MPI)||defined(__CHARMC__)
         if (domain.numSlices() != 1) {
            domain.planeElemIds[planeElems++] = zidx ;
         }
#endif
         for (Index_t col=0; col<heightElems; ++col) {
            Index_t *localNode = domain.nodelist(zidx) ;
            localNode[0] = nidx                                           ;
            localNode[1] = nidx                                       + 1 ;
            localNode[2] = nidx                         + heightNodes + 1 ;
            localNode[3] = nidx                         + heightNodes     ;
            localNode[4] = nidx + edgeNodes*heightNodes                   ;
            localNode[5] = nidx + edgeNodes*heightNodes               + 1 ;
            localNode[6] = nidx + edgeNodes*heightNodes + heightNodes + 1 ;
            localNode[7] = nidx + edgeNodes*heightNodes + heightNodes     ;
            ++zidx ;
            ++nidx ;
         }
         ++nidx ;
      }
      nidx += heightNodes ;
   }

   /* connect new plane to old plane */
   nidx = (coreNodes-1)*edgeNodes*heightNodes ;
   zidx = coreElems*edgeElems*heightElems ;
   for (Index_t row=0; row<(coreElems-1); ++row) {
#if defined(COEVP_MPI)||defined(__CHARMC__)
      if (domain.numSlices() != 1) {
         domain.planeElemIds[planeElems++] = zidx ;
      }
#endif
      for (Index_t col=0; col<heightElems; ++col) {
         Index_t *localNode = domain.nodelist(zidx) ;
         localNode[0] = nidx                                           ;
         localNode[1] = nidx                                       + 1 ;
         localNode[2] = nidx                         + heightNodes + 1 ;
         localNode[3] = nidx                         + heightNodes     ;
         localNode[4] = nidx + edgeNodes*heightNodes                   ;
         localNode[5] = nidx + edgeNodes*heightNodes               + 1 ;
         localNode[6] = nidx + edgeNodes*heightNodes + heightNodes + 1 ;
         localNode[7] = nidx + edgeNodes*heightNodes + heightNodes     ;
         ++zidx ;
         ++nidx ;
      }
      ++nidx ;
   }
   /* do work in Z wing */
   nidx = coreNodes*heightNodes*edgeNodes ;
   for (Index_t plane=coreElems+1; plane<edgeElems; ++plane) {
      for (Index_t row=0; row<(coreElems-1); ++row) {
#if defined(COEVP_MPI)||defined(__CHARMC__)
         if (domain.numSlices() != 1) {
            domain.planeElemIds[planeElems++] = zidx ;
         }
#endif
         for (Index_t col=0; col<heightElems; ++col) {
            Index_t *localNode = domain.nodelist(zidx) ;
            localNode[0] = nidx                                               ;
            localNode[1] = nidx                                           + 1 ;
            localNode[2] = nidx                             + heightNodes + 1 ;
            localNode[3] = nidx                             + heightNodes     ;
            localNode[4] = nidx + heightNodes*(coreNodes-1)                   ;
            localNode[5] = nidx + heightNodes*(coreNodes-1)               + 1 ;
            localNode[6] = nidx + heightNodes*(coreNodes-1) + heightNodes + 1 ;
            localNode[7] = nidx + heightNodes*(coreNodes-1) + heightNodes     ;
            ++zidx ;
            ++nidx ;
         }
         ++nidx ;
      }
      nidx += heightNodes ;
   }
   /* create the bridge elements between disjoint meshes */
   nidx = (coreNodes-1)*edgeNodes*heightNodes +
      (coreNodes-1)*heightNodes ;
   for (Index_t row=coreElems ; row<edgeElems; ++row) {
#if defined(COEVP_MPI)||defined(__CHARMC__)
      if (domain.numSlices() != 1) {
         domain.planeElemIds[planeElems++] = zidx ;
      }
#endif
      for (Index_t col=0; col<heightElems; ++col) {
         Index_t *localNode = domain.nodelist(zidx) ;
         localNode[0] = nidx                                             ;
         localNode[1] = nidx                                         + 1 ;
         localNode[2] = nidx                           + heightNodes + 1 ;
         localNode[3] = nidx                           + heightNodes     ;
         ++zidx ;
         ++nidx ;
      }
      ++nidx ;
   }
   zidx -= heightElems*wingElems ;
   nidx = (coreNodes-1)*edgeNodes*heightNodes +
      (coreNodes-1)*heightNodes - heightNodes ;
   for (Index_t col=0; col<heightElems; ++col) {
      Index_t *localNode = domain.nodelist(zidx) ;
      localNode[4] = nidx                               ;
      localNode[5] = nidx                           + 1 ;
      localNode[6] = nidx + edgeNodes*heightNodes   + 1 ;
      localNode[7] = nidx + edgeNodes*heightNodes       ;
      ++zidx ;
      ++nidx ;
   }
   ++nidx ;

   nidx = coreNodes*edgeNodes*heightNodes +
      (coreNodes-1)*heightNodes - heightNodes ;
   for (Index_t plane=coreElems+1; plane<edgeElems; ++plane) {
      for (Index_t col=0; col<heightElems; ++col) {
         Index_t *localNode = domain.nodelist(zidx) ;
         localNode[4] = nidx                                             ;
         localNode[5] = nidx                                         + 1 ;
         localNode[6] = nidx + (coreNodes-1)*heightNodes             + 1 ;
         localNode[7] = nidx + (coreNodes-1)*heightNodes                 ;
         ++zidx ;
         ++nidx ;
      }
      ++nidx ;
      nidx -= heightNodes ;
      nidx += (coreNodes-1)*heightNodes ;
   }

#if defined(COEVP_MPI)||defined(__CHARMC__)
   if (domain.numSlices() != 1) {
      if (planeElems != domain.commElems()) {
         printf("%d %d error computing comm elems\n",
               planeElems, domain.commElems()) ;
         exit(-1) ;
      }
   }
#endif

   /* Create a material IndexSet (entire domain same material for now) */
   for (Index_t i=0; i<domElems; ++i) {
      domain.matElemlist(i) = i ;
   }

   char name[100] ;
#if defined(COEVP_MPI)||defined(__CHARMC__)
   sprintf(name, "checkConn%d.sami", domain.sliceLoc()) ;
#else
   sprintf(name, "checkConn.sami") ;
#endif

   /* initialize material parameters */
   //   domain.dtfixed() = Real_t(-1.0e-7) ;
   //   domain.deltatime() = Real_t(1.0e-7) ;
   domain.dtfixed() = Real_t(-1.0e-9) ;
   domain.deltatime() = Real_t(1.0e-9) ;

   domain.deltatimemultlb() = Real_t(1.1) ;
   domain.deltatimemultub() = Real_t(1.2) ;
   domain.stoptime()  = domainStopTime;
   domain.dtcourant() = Real_t(1.0e+20) ;
   domain.dthydro()   = Real_t(1.0e+20) ;
   domain.dtmax()     = Real_t(1.0e-2) ;
   domain.time()    = Real_t(0.) ;
   domain.cycle()   = 0 ;
   if(simStopCycle != 0)
   {
	domain.stopcycle() = simStopCycle;
   }
   else
   {
	domain.stopcycle() = INT_MAX; 
   }

   domain.e_cut() = Real_t(1.0e-7) ;
   domain.p_cut() = Real_t(1.0e-7) ;
   domain.q_cut() = Real_t(1.0e-7) ;
   domain.u_cut() = Real_t(1.0e-7) ;
   domain.v_cut() = Real_t(1.0e-10) ;

   domain.crqt()        = Real_t(0.01) ;
   domain.hgcoef()      = Real_t(3.0) ;
   domain.ss4o3()       = Real_t(4.0)/Real_t(3.0) ;

   domain.qstop()              =  Real_t(1.0e+12) ;
   domain.monoq_max_slope()    =  Real_t(1.0) ;
   domain.monoq_limiter_mult() =  Real_t(2.0) ;
   domain.qlc_monoq()          = Real_t(0.5) ;
   domain.qqc_monoq()          = Real_t(2.0)/Real_t(3.0) ;
   domain.qqc()                = Real_t(2.0) ;

   domain.pmin() =  Real_t(0.) ;
   //   domain.emin() = Real_t(-1.0e+15) ;
   domain.emin() = Real_t(-1.0e+50) ;

   domain.dvovmax() =  Real_t(0.1) ;

   //   domain.eosvmax() =  Real_t(1.0e+9) ;
   domain.eosvmax() =  Real_t(1.0e+50) ;
   domain.eosvmin() =  Real_t(1.0e-9) ;

   domain.refdens() =  Real_t(1.664e1) ;  // g / cm^3

   /* initialize field data */
   for (Index_t i=0; i<domElems; ++i) {
      Real_t x_local[8], y_local[8], z_local[8] ;
      Index_t *elemToNode = domain.nodelist(i) ;
      for( Index_t lnode=0 ; lnode<8 ; ++lnode )
      {
         Index_t gnode = elemToNode[lnode];
         x_local[lnode] = domain.x(gnode);
         y_local[lnode] = domain.y(gnode);
         z_local[lnode] = domain.z(gnode);
      }

      // volume calculations
      Real_t volume = CalcElemVolume(x_local, y_local, z_local );
      domain.volo(i) = volume ;
      domain.elemMass(i) = volume ;
      for (Index_t j=0; j<8; ++j) {
         Index_t idx = elemToNode[j] ;
         domain.nodalMass(idx) += volume / Real_t(8.0) ;
      }
   }

#if defined(COEVP_MPI)
   CommRecv(&domain, MSG_COMM_SBN, 1, domain.commNodes()) ;
#endif

   for (Index_t i=0; i<domElems; ++i) {
      domain.e(i) = 0.;
   }

#if defined(COEVP_MPI)||defined(__CHARMC__)
   if (domain.sliceLoc() == 0)
#endif
   {
      for (int plane=0; plane<coreNodes; ++plane) {
         for (int row=0; row<edgeNodes; ++row) {
            domain.xd(row*heightNodes +
                  plane*edgeNodes*heightNodes) = Real_t(1.75e-2) ;
         }
      }
      for (int plane=0; plane<wingNodes; ++plane) {
         for (int row=0; row<(coreNodes-1); ++row) {
            domain.xd(coreNodes*edgeNodes*heightNodes +
                  row*heightNodes +
                  plane*(coreNodes-1)*heightNodes) = Real_t(1.75e-2) ;
         }
      }
   }

   /* set up symmetry nodesets */
   /* Z symmetry */
   for (Index_t i=0; i<edgeNodes*heightNodes; ++i) {
      domain.symmZ(i) = i ;
   }
   /* Y symmetry */
   nidx = 0 ;
   for (int plane=0;plane<coreNodes; ++plane) {
      for (int col=0; col<heightNodes; ++col) {
         domain.symmY(nidx++) = plane*edgeNodes*heightNodes + col ;
      }
   }
   for (int plane=0;plane<wingNodes; ++plane) {
      for (int col=0; col<heightNodes; ++col) {
         domain.symmY(nidx++) = coreNodes*edgeNodes*heightNodes +
            plane*(coreNodes-1)*heightNodes + col ;
      }
   }
   /* X Symmetry */
#if defined(COEVP_MPI)||defined(__CHARMC__)
   if (domain.sliceLoc() == 0) 
#endif
   {
      nidx = 0 ;
      for (int plane=0; plane<coreNodes; ++plane) {
         for (int row=0; row<edgeNodes; ++row) {
            domain.symmX(nidx++) = plane*edgeNodes*heightNodes +
               row*heightNodes ;
         }
      }
      for (int plane=0; plane<wingNodes; ++plane) {
         for (int row=0; row<(coreNodes-1); ++row) {
            domain.symmX(nidx++) = coreNodes*edgeNodes*heightNodes +
               plane*(coreNodes-1)*heightNodes +
               row*heightNodes ;
         }
      }
   }

   /* set up elemement connectivity information */
   domain.lxim(0) = 0 ;
   for (Index_t i=1; i<domElems; ++i) {
      domain.lxim(i)   = i-1 ;
      domain.lxip(i-1) = i ;
   }
   domain.lxip(domElems-1) = domElems-1 ;

   for (Index_t i=0; i<heightElems; ++i) {
      /* These are unused dummy values at boundaries */
      /* They are initialized for Visualization purposes only */
      domain.letam(i) = i ; 
#if 0
      domain.letap(domElems-heightElems+i) = domElems-heightElems+i ;
#endif
   }
   for (Index_t i=heightElems; i<domElems; ++i) {
      domain.letam(i) = i-heightElems ;
      domain.letap(i-heightElems) = i ;
   }

   /* Patch letap,letam for notch plane of Elements */

   /* First, patch letap to connect Z wing to Y wing */
   for (int i=domElems - wingElems*heightElems; i<domElems; ++i) {
      domain.letap(i) = coreElems*edgeElems*heightElems - (domElems - i) ;
   }
   /* Adjust the connectivity of the notch plane and the Z wing */
   for (int plane=0; plane<wingElems; ++plane) {
      for (int col=0; col<heightElems; ++col) {
         domain.letam(coreElems*edgeElems*heightElems +
               (coreElems-1)*heightElems*wingElems +
               plane*heightElems + col) = 
            coreElems*edgeElems*heightElems +
            (coreElems-1)*heightElems - heightElems +
            plane*(coreElems-1)*heightElems + col ;

         domain.letap(coreElems*edgeElems*heightElems +
               (coreElems-1)*heightElems - heightElems +
               plane*(coreElems-1)*heightElems + col)  =
            coreElems*edgeElems*heightElems +
            (coreElems-1)*heightElems*wingElems +
            plane*heightElems + col ;
      }
   }

   /* Create connectivity for lzetam, lzetap */
#ifndef OLD_LZETA_CONNECTIVITY
   for (Index_t i=0; i<edgeElems*heightElems; ++i) {
      /* these are dummmy values for visualization only */
      domain.lzetam(i) = i ;
#if 0
      domain.lzetap(domElems-edgeElems*heightElems+i) =
         domElems-edgeElems*heightElems+i ;
#endif
   }
   for (Index_t i=edgeElems*heightElems;
         i<coreElems*edgeElems*heightElems+(coreElems-1)*heightElems; ++i) {
      domain.lzetam(i) = i - edgeElems*heightElems ;
      domain.lzetap(i-edgeElems*heightElems) = i ;
   }
   /* patch lzetap to connect Y wing to Z wing */
   for (int i=domElems - wingElems*heightElems; i<domElems; ++i) {
      domain.lzetap(coreElems*edgeElems*heightElems - (domElems - i)) = i ;
   }
   /* set lzetam and lzetap for Z wing elements, minus notch plane */
   for (int i=coreElems*edgeElems*heightElems;
         i < coreElems*edgeElems*heightElems+(coreElems-1)*heightElems;
         ++i) {
      domain.lzetap(i) = i + (coreElems-1)*heightElems ;
   }
   for (int i=coreElems*edgeElems*heightElems+(coreElems-1)*heightElems;
         i<coreElems*edgeElems*heightElems + (coreElems-1)*wingElems*heightElems;
         ++i) {
      domain.lzetam(i) = i - (coreElems-1)*heightElems ;
      domain.lzetap(i) = i + (coreElems-1)*heightElems ;
   }
   /* set lzetam, lzetap for notch plane */
   for (int i=domElems-wingElems*heightElems; i<domElems; ++i) {
      if (i >= domElems-wingElems*heightElems + heightElems) {
         domain.lzetam(i) = i - heightElems ;
      }
      domain.lzetap(i) = i + heightElems ;
   }
   /* patch lzetap for notch plane row of elements */
   for (int col=0; col<heightElems; ++col) {
      domain.lzetap(coreElems*edgeElems*heightElems -
            wingElems*heightElems - heightElems + col) =
         domElems - wingElems*heightElems + col ;
      domain.lzetam(domElems - wingElems*heightElems + col) =
         coreElems*edgeElems*heightElems - wingElems*heightElems -
         heightElems + col ;
   }
#else
   for (Index_t i=0; i<edgeElems*heightElems; ++i) {
      domain.lzetam(i) = i ;
      domain.lzetap(domElems-edgeElems*heightElems+i) =
         domElems-edgeElems*heightElems+i ;
   }
   for (Index_t i=edgeElems*heightElems;
         i<coreElems*edgeElems*heightElems; ++i) {
      domain.lzetam(i) = i - edgeElems*heightElems ;
      domain.lzetap(i-edgeElems*heightElems) = i ;
   }
   /* patch lzetap to connect Y wing to Z wing */
   for (int i=domElems - wingElems*heightElems; i<domElems; ++i) {
      domain.lzetap(coreElems*edgeElems*heightElems - (domElems - i)) = i ;
   }
   /* set lzetam and lzetap for Z wing elements, minus notch plane */
   for (int i=coreElems*edgeElems*heightElems;
         i<coreElems*edgeElems*heightElems + (coreElems-1)*wingElems*heightElems;
         ++i) {
      domain.lzetam(i) = i - (coreElems-1)*heightElems ;
      domain.lzetap(i) = i + (coreElems-1)*heightElems ;
   }
   /* patch lzetam */
   for (int i=coreElems*edgeElems*heightElems;
         i<coreElems*edgeElems*heightElems + (coreElems-1)*heightElems; ++i) {
      domain.lzetam(i) = i - edgeElems*heightElems ;
   }
   /* set lzetam, lzetap for notch plane */
   for (int i=domElems-wingElems*heightElems; i<domElems; ++i) {
      domain.lzetam(i) = i - heightElems ;
      domain.lzetap(i) = i + heightElems ;
   }
   /* patch lzetap for notch plane row of elements */
   for (int col=0; col<heightElems; ++col) {
      domain.lzetap(coreElems*edgeElems*heightElems -
            wingElems*heightElems - heightElems + col) =
         domElems - wingElems*heightElems + col ;
      domain.lzetam(domElems - wingElems*heightElems + col) =
         coreElems*edgeElems*heightElems - wingElems*heightElems -
         heightElems + col ;
   }
#endif

   /* set up boundary condition information */
   for (Index_t i=0; i<domElems; ++i) {
      domain.elemBC(i) = 0 ;  /* clear BCs by default */
   }

#if defined(COEVP_MPI)||defined(__CHARMC__)

   if (domain.numSlices() > 1) {
      if (domain.sliceLoc() == 0) {
         for (int i=0; i<domain.commElems(); ++i) {
            /* adjust lxip() to point at (end of) com buffer data */
            domain.lxip(domain.planeElemIds[i]+domain.sliceHeight()-1)
               = domElems+i;
         }
      }
      else if (domain.sliceLoc() == domain.numSlices()-1) {
         /* adjust lxim() to point at (end of) comm buffer data */
         for (int i=0; i<domain.commElems(); ++i) {
            domain.lxim(domain.planeElemIds[i]) = domElems+i ;
         }
      }
      else /* two messages recevied */ {
         /* "plane below" data goes at end of buffer */
         int endOfBuffer = domElems ;
         for (int i=0; i<domain.commElems(); ++i) {
            domain.lxim(domain.planeElemIds[i]) = endOfBuffer+i ;
         }
         /* "plane above" data goes after "plane below" data */
         endOfBuffer += domain.commElems() ;
         for (int i=0; i<domain.commElems(); ++i) {
            /* adjust lxip() to point at (end of) com buffer data */
            domain.lxip(domain.planeElemIds[i]+domain.sliceHeight()-1)
               = endOfBuffer+i;
         }
      }
   }

#endif

   /* faces on "external" boundaries will be */
   /* symmetry plane or free surface BCs */
   for (int i=0; i<edgeElems*heightElems; ++i) {
      domain.elemBC(i) |= ZETA_M_SYMM ;
   }
   for (int i=domElems-heightElems; i<domElems; ++i) {
      domain.elemBC(i) |= ZETA_P_FREE ;
   }
   for (int i= domElems - wingElems*heightElems - (coreElems-1)*heightElems;
         i<domElems - wingElems*heightElems; ++i) {
      domain.elemBC(i) |= ZETA_P_FREE ;
   }

   for (int plane=0; plane<coreElems; ++plane) {
      for (int col=0; col<heightElems; ++col) {
         domain.elemBC(plane*edgeElems*heightElems + col) |= ETA_M_SYMM ;
         domain.elemBC(plane*edgeElems*heightElems +
               (edgeElems-1)*heightElems +col) |= ETA_P_FREE ;
      }
   }
   for (int plane=0; plane <wingElems; ++plane) {
      for (int col=0; col<heightElems; ++col) {
         domain.elemBC(coreElems*edgeElems*heightElems +
               plane*(coreElems-1)*heightElems + col) |= ETA_M_SYMM ;
      }
   }


   for (int plane=0; plane<coreElems; ++plane) {
      for (int row=0; row<edgeElems; ++row) {
         domain.elemBC(plane*edgeElems*heightElems + row*heightElems) |=
#if defined(COEVP_MPI)||defined(__CHARMC__)
		 ((domain.sliceLoc() == 0) ? XI_M_SYMM : XI_M_COMM ) ;
#else
         XI_M_SYMM ;	
#endif
         domain.elemBC(plane*edgeElems*heightElems +

                        row*heightElems + heightElems-1) |= 
#if defined(COEVP_MPI)||defined(__CHARMC__)
                           ((domain.sliceLoc() == domain.numSlices()-1) ? XI_P_FREE : XI_P_COMM ) ;
#else
         XI_P_FREE ;
#endif
      }
   }
   for (int plane=0; plane<wingElems; ++plane) {
      for (int row=0; row<(coreElems-1); ++row) {
         domain.elemBC(coreElems*edgeElems*heightElems +
                       plane*(coreElems-1)*heightElems + row*heightElems) |= 
#if defined(COEVP_MPI)||defined(__CHARMC__)
		 ((domain.sliceLoc() == 0) ? XI_M_SYMM : XI_M_COMM ) ;
#else
         XI_M_SYMM ;
#endif
         domain.elemBC(coreElems*edgeElems*heightElems +
                       plane*(coreElems-1)*heightElems +
                       row*heightElems + heightElems-1) |=
#if defined(COEVP_MPI)||defined(__CHARMC__)
                          ((domain.sliceLoc() == domain.numSlices()-1) ? XI_P_FREE : XI_P_COMM ) ;
#else
         XI_P_FREE ;
#endif
      }
      domain.elemBC(coreElems*edgeElems*heightElems +
                    wingElems*(coreElems-1)*heightElems + plane*heightElems) |= 
#if defined(COEVP_MPI)||defined(__CHARMC__)
	      ((domain.sliceLoc() == 0) ? XI_M_SYMM : XI_M_COMM ) ;
#else
      XI_M_SYMM ;
#endif
      domain.elemBC(coreElems*edgeElems*heightElems +
                    wingElems*(coreElems-1)*heightElems +
                    plane*heightElems + heightElems-1) |=
#if defined(COEVP_MPI)||defined(__CHARMC__)
                       ((domain.sliceLoc() == domain.numSlices()-1) ? XI_P_FREE : XI_P_COMM ) ;
#else
      XI_P_FREE ;
#endif

   }

#if 0
   {
      int myRank = 0;
      int numRanks = 1;
      DumpDomain(&domain, myRank, numRanks) ;
   }
   exit(0) ;
#endif

}


void Lulesh::ConstructFineScaleModel(bool sampling, ModelDatabase * global_modelDB, ApproxNearestNeighbors* global_ann, ApproxNearestNeighborsDB **global_anndb, int flanning, int flann_n_trees, int flann_n_checks, int global_ns, int nnonly, int use_vpsc, double c_scaling, hg_service_mode mode, ssg_t ssg, margo_instance_id mid)
{
   Index_t domElems = domain.numElem();

   ConstitutiveGlobal cm_global;
#ifdef _OPENMP
#pragma omp parallel for
#endif
   for (Index_t i=0; i<domElems; ++i) {

      Plasticity* plasticity_model;

      if (use_vpsc == 1) {
         // New vpsc inititialization
         plasticity_model = (vpsc*) (new vpsc(c_scaling));
      } else {
         // Old Taylor initialization
         //
         // Construct the fine-scale plasticity model
         double D_0 = 1.e-2;
         double m = 1./20.;
         double g = 2.e-3; // (Mbar)
         //      double m = 1./2.;
         //      double g = 1.e-4; // (Mbar) Gives a reasonable looking result for m = 1./2.
         //      double m = 1.;
         //      double g = 2.e-6; // (Mbar) Gives a reasonable looking result for m = 1.

         plasticity_model = (Plasticity*)(new Taylor(D_0, m, g));
      }

      // Construct the equation of state
      EOS* eos_model;
#if 1
      {
         /* From Table 1 (converted from GPa to Mbar) in P. J. Maudlin et al.,
            "On the modeling of the Taylor cylinder impact test for orthotropic
            textured materials: experiments and simulations", Inter. J.
            Plasticity 15 (1999), pp. 139-166.
            */
         double k1 = 1.968;  // Mbar
         double k2 = 2.598;  // Mbar
         double k3 = 2.566;  // Mbar
         double Gamma = 1.60;  // dimensionless
         eos_model = (EOS*)(new MieGruneisen(k1, k2, k3, Gamma));
      }
#else
      {
         /* Bulk pressure model from N. Barton, "Cold Energy Integration",
            UCRL-TR-220933 (2006).  Tantalum parameters from private communication.
            */

         double K0 = 1.94;      // Mbar
         double a  = 0.42;      // dimensionless
         double S  = 1.2;       // dimensionless
         double Gamma0 = 1.67;  // dimensionless
         eos_model = (EOS*)(new BulkPressure(K0, a, S, Gamma0));
      }
#endif

      // Construct the constitutive model
      double bulk_modulus = 1.94; // Tantallum (Mbar)
      double shear_modulus = 6.9e-1; // Tantallum (Mbar)
      {
         Real_t B[3][8] ; /** shape function derivatives */
         Real_t D[6] ;
         Real_t W[3] ;
         Real_t x_local[8] ;
         Real_t y_local[8] ;
         Real_t z_local[8] ;
         Real_t xd_local[8] ;
         Real_t yd_local[8] ;
         Real_t zd_local[8] ;
         Real_t detJ = Real_t(0.0) ;

         const Index_t* const elemToNode = domain.nodelist(i) ;

         // get nodal coordinates from global arrays and copy into local arrays.
         for( Index_t lnode=0 ; lnode<8 ; ++lnode )
         {
            Index_t gnode = elemToNode[lnode];
            x_local[lnode] = domain.x(gnode);
            y_local[lnode] = domain.y(gnode);
            z_local[lnode] = domain.z(gnode);
         }

         // get nodal velocities from global array and copy into local arrays.
         for( Index_t lnode=0 ; lnode<8 ; ++lnode )
         {
            Index_t gnode = elemToNode[lnode];
            xd_local[lnode] = domain.xd(gnode);
            yd_local[lnode] = domain.yd(gnode);
            zd_local[lnode] = domain.zd(gnode);
         }

         // compute the velocity gradient at the new time (i.e., before the
         // nodal positions get backed up a half step below).  Question:
         // where are the velocities centered at this point?

         CalcElemShapeFunctionDerivatives( x_local,
               y_local,
               z_local,
               B, &detJ );

         CalcElemVelocityGradient( xd_local,
               yd_local,
               zd_local,
               B, detJ, D, W );

         Tensor2Gen L;

         L(1,1) = D[0];         // dxddx
         L(1,2) = D[5] - W[2];  // dyddx
         L(1,3) = D[4] + W[1];  // dzddx
         L(2,1) = D[5] + W[2];  // dxddy 
         L(2,2) = D[1];         // dyddy
         L(2,3) = D[3] - W[0];  // dzddy
         L(3,1) = D[4] - W[1];  // dxddz
         L(3,2) = D[3] + W[0];  // dyddz
         L(3,3) = D[2];         // dzddz

         int point_dimension = plasticity_model->pointDimension();
         ApproxNearestNeighbors* ann;
         ModelDatabase *modelDB;
         if (global_modelDB) {
            modelDB = global_modelDB;
         } else {
            modelDB = new ModelDB_HashMap();
         }

         if (nnonly) {
           if (*global_anndb) {
             anndb = *global_anndb;
           }
           else {
             // should be checked for in lulesh_main
             assert(flanning);
             assert(mode == HGSVC_NONE || mode == HGSVC_NNONLY);
             if (mode == HGSVC_NONE) {
#ifdef FLANN
               anndb = new ApproxNearestNeighborsFLANNDB(point_dimension, flann_n_trees, flann_n_checks, false);
#else
               throw std::runtime_error("FLANN not compiled in");
#endif
             }
             else {
               anndb = new ApproxNearestNeighborsDBHGWrapClient(ssg, mid, false);
             }
             if (global_ns && !*global_anndb) *global_anndb = anndb;
           }
         }
         else {
           if (global_ann) {
             ann = global_ann;
           }
           else {
             if (flanning) {
#ifdef FLANN
               ann = (ApproxNearestNeighbors*)(new ApproxNearestNeighborsFLANN(point_dimension, flann_n_trees, flann_n_checks));
#else
               throw std::runtime_error("FLANN not compiled in"); 
#endif
            } else {
               std::string mtreeDirectoryName = ".";
               ann = (ApproxNearestNeighbors*)(new ApproxNearestNeighborsMTree(point_dimension,
                        "kriging_model_database",
                        mtreeDirectoryName,
                        &(std::cout),
                        false));
            }
         }
         if ( global_ns && !global_ann){// only true for 1st element
            global_ann=ann; 
         }

         size_t state_size;
         domain.cm(i) = (Constitutive*)(new ElastoViscoPlasticity(cm_global, ann, modelDB, nullptr, L, bulk_modulus, shear_modulus, eos_model,
                  plasticity_model, sampling, state_size));
         domain.cm_state(i) = operator new(state_size);
         domain.cm(i)->getState(domain.cm_state(i));
      }
   }

#ifdef WRITE_FSM_EVAL_COUNT
   // Set the element number at which the fine-scale model evaluation count is recorded
   Index_t fsm_count_elem = 0;

   ostringstream fsm_count_filename;
   fsm_count_filename << "fsm_count_" << fsm_count_elem;
   ofstream fsm_count_file(fsm_count_filename.str().c_str());

   {
      // Print the location of the element center at which the fine-scale model evaluation 
      // count is being recorded

      Index_t *localNode = domain.nodelist(fsm_count_elem) ;
      Real_t xav, yav, zav;
      xav = yav = zav = Real_t(0.) ;
      for (Int_t i=0; i<8; ++i) {
         xav += domain.x(localNode[i]) ;
         yav += domain.y(localNode[i]) ;
         zav += domain.z(localNode[i]) ;
      }
      xav /= Real_t(8.) ;
      yav /= Real_t(8.) ;
      zav /= Real_t(8.) ;

      cout << "Plotting fine-scale model evaluations in element " << fsm_count_elem
         << " at (" << xav << "," << yav << "," << zav << ")" << endl;
   }

   printf("Fine scale models initialized\n");
   fflush(stdout);

   Int_t cumulative_fsm_count = 0;
#endif   

}

void Lulesh::ExchangeNodalMass()
{
#if defined(COEVP_MPI)
   Real_t *fieldData = &domain.nodalMass(0) ;
   CommSend(&domain, MSG_COMM_SBN, 1, &fieldData,
         domain.planeNodeIds, domain.commNodes(), domain.sliceHeight()) ;
   CommSBN(&domain, 1, &fieldData,
         domain.planeNodeIds, domain.commNodes(), domain.sliceHeight()) ;
#endif


#if 0
   MPI_Finalize() ;
   exit(0) ;
#endif
}

void Lulesh::go(int myRank, int numRanks, int sampling, int visit_data_interval,int file_parts, int debug_topology)
{

#if defined(LOGGER)   // did I mention that I hate this define stuff?
   Logger  &logger = Locator::getLogger();
#endif

   /* timestep to solution */
   while(domain.time() < domain.stoptime() and domain.cycle() < domain.stopcycle()) {
#if defined(LOGGER)
      logger.logStartTimer("outer");
#endif
#ifdef SILO
      char meshName[64] ;
      if ((visit_data_interval !=0) && (domain.cycle() % visit_data_interval == 0)) {
         DumpDomain(&domain, domain.sliceLoc(), domain.numSlices(),
               ((domain.numSlices() == 1) ? file_parts : 0), sampling, debug_topology ) ;
      }
#endif
      TimeIncrement() ;
      LagrangeLeapFrog() ;
      /* problem->commNodes->Transfer(CommNodes::syncposvel) ; */
      int maxIters = UpdateStressForElems();
      UpdateStressForElems2(maxIters);
#ifdef LULESH_SHOW_PROGRESS
      //      printf("time = %e, dt=%e\n",
      //             double(domain.time()), double(domain.deltatime()) ) ;
#if defined(COEVP_MPI)||defined(__CHARMC__)
      if (domain.sliceLoc() == 0) 
#endif

      {
         printf("step = %d, time = %e, dt=%e\n",
               domain.cycle(), double(domain.time()), double(domain.deltatime()) ) ;
         fflush(stdout);
      }

#ifdef PRINT_PERFORMANCE_DIAGNOSTICS
      if ( sampling ) {

         int total_samples = 0;
         int total_interpolations = 0;
         Index_t domElems = domain.numElem() ;
         for (int i=0; i<domElems; ++i) {
            total_samples += domain.cm(i)->getNumSamples();
            total_interpolations += domain.cm(i)->getNumSuccessfulInterpolations();
         }

         cout << "   Interpolation efficiency = " << (double)total_interpolations / (double)total_samples << endl;

      }
#endif
#endif

#ifdef WRITE_FSM_EVAL_COUNT
      if ( sampling ) {
         Int_t num_fsm_evals = domain.cm(fsm_count_elem)->getNumSamples()
            - domain.cm(fsm_count_elem)->getNumSuccessfulInterpolations() ;
         fsm_count_file << domain.time() << "  " << num_fsm_evals - cumulative_fsm_count << endl;
         cumulative_fsm_count = num_fsm_evals;
      }
#endif
#if defined(LOGGER)
      logger.logIncrTimer("outer");
      logger.incrTimeStep();
#endif
   }  /* while */

	FinalTime();

#ifdef WRITE_FSM_EVAL_COUNT
   fsm_count_file.close();
#endif   
   if ( sampling ) {

      const int num_stats = domain.cm(0)->getNumberStatistics();
      std::vector<std::string> stat_strs = domain.cm(0)->getStatisticsNames();
      auto stats     = std::make_unique<double[]>(num_stats);
      auto agg_stats = std::make_unique<double[]>(num_stats);
      auto max_stats = std::make_unique<double[]>(num_stats);
      auto min_stats = std::make_unique<double[]>(num_stats);
      std::fill_n(agg_stats.get(), num_stats, 0.);
      std::fill_n(max_stats.get(), num_stats, std::numeric_limits<double>::lowest());
      std::fill_n(min_stats.get(), num_stats, std::numeric_limits<double>::max());

      Real_t point_average = Real_t(0.);
      Real_t value_average = Real_t(0.);
      Real_t point_max = Real_t(0.);
      Real_t value_max = Real_t(0.);
      Index_t domElems = domain.numElem() ;
      for (Index_t i=0; i<domElems; ++i) {

         domain.cm(i)->getStatistics(stats.get(), num_stats);
         for (int j = 0; j < num_stats; j++) {
           agg_stats[j] += stats[j];
           max_stats[j] = std::max(stats[j], max_stats[j]);
           min_stats[j] = std::min(stats[j], min_stats[j]);
         }

         Real_t point_norm = domain.cm(i)->getAveragePointNorm();
         Real_t value_norm = domain.cm(i)->getAverageValueNorm();
         Real_t point_norm_max = domain.cm(i)->getPointNormMax();
         Real_t value_norm_max = domain.cm(i)->getValueNormMax();

         point_average += point_norm;
         value_average += value_norm;

         if ( point_norm_max > point_max ) {
            point_max = point_norm_max;
         }
         if ( value_norm_max > value_max ) {
            value_max = value_norm_max;
         }
      }
      averageNumModels /= domElems;
      averageNumPairs /= domElems;
      point_average /= domElems;
      value_average /= domElems;

      for (int i = 0; i < num_stats; i++) {
        cout << stat_strs[i]            << ": "
             << agg_stats[i]            << " (total), "
             << agg_stats[i] / domElems << " (average), "
             << min_stats[i]            << " (min), "
             << max_stats[i]            << " (max)" << endl;
      }

      cout << "Scaled query average = " << point_average << ", max = " << point_max << endl;
      cout << "Scaled value average = " << value_average << ", max = " << value_max << endl; 


   }

#ifdef SILO
   if ((visit_data_interval != 0) && (domain.cycle() % visit_data_interval != 0)) {
      DumpDomain(&domain, domain.sliceLoc(), domain.numSlices(), 
            ((domain.numSlices() == 1) ? file_parts : 0), sampling, debug_topology ) ;
   }
#endif

#ifdef WRITE_CHECKPOINT

   // Write a checkpoint file for testing

   ofstream checkpoint_file("checkpoint");
   checkpoint_file.setf(ios::scientific);
   checkpoint_file.precision(13);

   for (Index_t i=0; i<domain.numNode(); ++i) {
      checkpoint_file << i << " " << domain.x(i) << " " << domain.y(i) << " " << domain.z(i)
         << " " << domain.xd(i) << " " << domain.yd(i) << " " << domain.zd(i) << endl;
   }

   checkpoint_file.close();
#endif
}

Lulesh::~Lulesh()
{
  for (Index_t i = 0; i < domain.numElem(); i++) delete domain.cm(i);
}
