##############################################################
#
#  GENETIC TEMPERING ON A GRAPHICS PROCESSING UNIT (GT-GPU)
#
##############################################################
# Made by Thomas E. Baker (2018)
# See accompanying license with this program
# This code is native to the julia programming language (v0.6.1)

#comments are limited here since some compilers don't easily see them in the wrapped OpenCL code
#Generates one Metropolis-Hastings iteration on one GPU thread

using OpenCL

const MCloop_kernel = "
   __kernel void MCloop(__global char *psi,
                     __global const int *intvec,
                     __global const int *randpos,
                     __global const float *randNum,
                     __global float *result)
    {

      char dE;
      const char charNegOne = -1;

      const int Nloop = intvec[0];
      const int latsize = intvec[1];
      const int GPUloops = intvec[2];
      const int L = intvec[3];
      const int J = intvec[4];

      const int gid = get_global_id(0);

      int ind,leftind,rightind,topind,bottomind;
      int count = 0;

      float currE = 0.;
      float currM = 0.;

      for (int i = 1;i<=latsize;i++)
      {
        ind = i;
        rightind = ind+L > latsize ? ind+L-latsize : ind+L;
        topind = (ind+1) % L == 1 ? ind+1-L : ind+1;
        currE -= convert_float(psi[gid + GPUloops * (ind-1)]*(psi[gid + GPUloops * (rightind-1)] + psi[gid + GPUloops * (topind-1)]));
        currM += convert_float(psi[gid + GPUloops * (ind-1)]);
      }

      float avgE = 0.;
      float avgM = 0.;
      float avgAbsM = 0.;
      float avgEsq = 0.;
      float avgMsq = 0.;
      float avgMfour = 0.;

      bool toggle;

      for (int i = 0;i < Nloop;i++)
      {
        ind = randpos[gid + GPUloops*i];

        leftind = ind-L < 1 ? ind-L+latsize : ind-L;
        rightind = ind+L > latsize ? ind+L-latsize : ind+L;
        topind = (ind+1) % L == 1 ? ind+1-L : ind+1;
        bottomind = ind % L == 1 ? ind-1+L : ind-1;

dE = psi[gid + GPUloops * (ind-1)]*(psi[gid + GPUloops * (leftind-1)]
                                + psi[gid + GPUloops * (rightind-1)]
                                + psi[gid + GPUloops * (topind-1)]
                                + psi[gid + GPUloops * (bottomind-1)]);

        toggle = false;
        if (dE < 0)
        {
          toggle = true;
        }
        else
        {
          if (randNum[gid + GPUloops*i] > convert_float(dE))
          {
            toggle = true;
          }
        }

      if (toggle)
      {
        psi[gid + GPUloops * (ind-1)] *= charNegOne;
        currE += 2.*convert_float(dE);
        currM += 2.*convert_float(psi[gid + GPUloops * (ind-1)]);

        avgE += currE;
        avgEsq += currE*currE;
        avgM += currM;
        avgMsq += currM*currM;
        avgAbsM += fabs(currM);
        avgMfour += currM*currM*currM*currM;

        count += 1;
      }
    }

    if (count > 0)
    {
      float fcount = convert_float(count);
      result[gid + GPUloops*0] = avgE;
      result[gid + GPUloops*1] = avgEsq;
      result[gid + GPUloops*2] = avgM;
      result[gid + GPUloops*3] = avgMsq;
      result[gid + GPUloops*4] = avgAbsM;
      result[gid + GPUloops*5] = avgMfour;
      result[gid + GPUloops*6] = fcount;
    }
    else
    {
      float origMsq,origMfour;
      origMsq = currM*currM;
      origMfour = origMsq*origMsq;
      result[gid + GPUloops*0] = currE;
      result[gid + GPUloops*1] = currE*currE;
      result[gid + GPUloops*2] = currM;
      result[gid + GPUloops*3] = origMsq;
      result[gid + GPUloops*4] = fabs(currM);
      result[gid + GPUloops*5] = origMfour;
      result[gid + GPUloops*6] = 0.;
    }
  }
"
