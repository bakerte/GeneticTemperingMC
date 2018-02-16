##############################################################
#
#  GENETIC TEMPERING ON A GRAPHICS PROCESSING UNIT (GT-GPU)
#
##############################################################
# Made by Thomas E. Baker (2018)
# See accompanying license with this program
# This code is native to the julia programming language (v0.6.1)
include("GPUpart.jl")

const L = 16 #lattice size on one side
const latsize = L*L #total number of sites...can be extended to non-square lattices
const J = 1 #1 is ferromagnetic

const Nloop = 1_000 #number of MC loops
const GPUloops = 1_000 #Number of instances of lattices on the GPU...an upper limit depends on the GPU's memory size

#adding a multiplier doesn't add much overhead, better to have higher M and lower Nloop on a laptop for heat reasons
#number of cycles for hysteresis loop
const delayloops = 5 #number of loops to check while thermalizing in generating hysteresis loop
const delaymult = 1 #store data for delaymult*Nloop iterations
#number of cycles in the final sweep
const relaxloops = 10 #delay on taking final sweep data
const maxmult = 10 #total Monte Carlo iterations: maxmult*Nloop iterations

const var = 1 #"1" = energy

const go_up = true #increase temperature or decrease?
const ordered = true #start from an ordered or a random state?
#####
const Tmin = 0.5 #2.25 #minimum temperature
const Tmax = 5.0 #2.35 #maximum temperature
const DeltaT = 0.1 #0.001 #temperature step
const numT = round(Int32,(Tmax-Tmin)/DeltaT)+1

#scan over a specific window...will add these points to the run schedule without duplicates
const Tc = true
const Tcmin = 2.2
const Tcmax = 2.3
const DeltaTc = 0.01
const numTc = round(Int32,(Tcmax-Tcmin)/DeltaTc)+1

result_buff=0

#information about the number of calcualtions for each lattice...can be removed
numcalcs =  Nloop*(2*delaymult+maxmult)*GPUloops #delay > Nloop ? Nloop*maxmult*GPUloops  : GPUloops*(Nloop*maxmult-delay)
println("single-thread equivalent iterations (per temperature): ",(@sprintf "%.3E" numcalcs))
println("Ideally, about ",floor(Int64,numcalcs/latsize)," lattice sites will be visited over ",GPUloops," threads and ",3," sweeps (each lattice site is visited about ",floor(Int64,numcalcs/latsize/GPUloops)," times per thread)")

#1536MB 1.2288E10
#4GB 3.2e+10
#8GB 6.4e+10
percentused = (2*4*GPUloops*Nloop)/1.2288E10
println("approximate upper limit of 1536MB GPU: ",(2*4*GPUloops*Nloop)/(8E6)," MB of 1536 MB (using ",GPUloops*Nloop," samples or ",100*percentused,"% of the total)")

#define functions for averages and standard deviation
avg(x::Array{Float64,1}...) = Float64[sum(x[p])/size(x[p],1) for p = 1:size(x,1)]
stDevsq(avgx::Array{Float64,1},x::Array{Float64,1}...) = Float64[sum(a->(x[p][a]-avgx[p])^2,1:size(x[p],1))/(size(x[p],1)*(size(x[p],1)-1)) for p = 1:size(x,1)]

#determines initial wavefunction psi across all GPU threads
if ordered
  psi = ones(Int8,GPUloops*latsize)
  else
  psi = Int8[(rand() > 0.5 ? Int8(1) : Int8(-1)) for j = 1:GPUloops*latsize]
end

device, ctx, queue = cl.create_compute_context() #creates a "context" for the GPU
psi_buff = cl.Buffer(Int8, ctx, (:rw, :copy), hostbuf=psi) #buffer wavefunction to GPU (can be written to and read from)
const intvec = Int32[Nloop,latsize,GPUloops,L,J]
intvec_buff = cl.Buffer(Int32, ctx, (:r, :copy), hostbuf=intvec)

#printed out data columns
println("#T       <E>     +/-E      <M>     +/-M      <|M|>   +/-|M|    C_v    +/-C_v   chi_|M|   +/-chi     B      +/-B")

#generates list of temperatures to run through
incrT = (go_up ? (Tmin:DeltaT:Tmax) : (Tmax:-DeltaT:Tmin))
incrTc = (go_up ? (Tcmin:DeltaTc:Tcmax) : (Tcmax:-DeltaTc:Tcmin))
runs = Float64[T for T = incrT]
Tc ? runs = sort(unique(vcat(runs,[T for T = incrTc]))) : 0.
go_up ? 0. : runs = reverse(runs)
const nruns = size(runs,1)
const interval = -nruns:nruns

data = zeros(Float64,Tc ? numTc : nruns,13) #output data
data[:,1] = Tc ? [T for T = (go_up ? (Tcmin:DeltaTc:Tcmax) : (Tcmax:-DeltaTc:Tcmin))] : runs
datafore = zeros(Float64,nruns,13) #output data going up in temperature
databack = zeros(Float64,nruns,13) #output data going down in temperature
datafore[:,1] = databack[:,1] = runs

savedpsi=Array{Int8,2}[zeros(Int8,GPUloops,latsize) for p = 1:nruns]#Array{Array{Int8,2}}(numT)
savedE=Array{Float64,1}[zeros(Float64,GPUloops) for p = 1:nruns]#Array{Array{Float64,1}}(numT)

geneticpsi=deepcopy(savedpsi) #saves wavefunctions for the final sweep

function dataline(dataArray::Array{Float64,2},m::Int64,t2::Float64,t1::Float64,gputime::Float64)
    println("T = ",dataArray[m,1],", loop time = ",t2-t1,", total GPU time = ",gputime)
    print((@sprintf "%.4f" dataArray[m,1])," ")
    for k = 2:2:size(dataArray,2)
      print((@sprintf "%.6f" dataArray[m,k])," ")
      print((@sprintf "%.6f" sqrt(dataArray[m,k+1]))," ")
    end
    println()
end

function savedata(dataArray::Array{Float64,2},m::Int64,t2::Float64,t1::Float64,gputime::Float64,values::Array{Float64,1},errors::Array{Float64,1})
  sizedatArray = size(dataArray,2)
  dataArray[m,2:2:sizedatArray-2] = deepcopy(values[1:size(values,1)-1])/latsize
  dataArray[m,3:2:sizedatArray-2] = deepcopy(errors[1:size(errors,1)-1])/latsize^2
  dataArray[m,sizedatArray-1:sizedatArray] = [values[size(values,1)],errors[size(errors,1)]]
  dataline(dataArray,m,t2,t1,gputime)
end

floatconvert(fcount::Float64,x::Array{Float32,1}) = [Float64(x[p])/fcount for p = 1:size(x,1)]

#start sweeps through temperature
counterloop2 = 0
breaksweep = false
@time for sweep = 1:2 #sweep loop: forward, backward, final loop
  breaksweep ? break : 0
for qq = interval
  if qq == 0
    if sweep == 2
      breaksweep = true
      break
    end
    continue
  end
  m = qq <= 0 ? nruns - abs(qq+1) : nruns + 1 - qq #site index
  temp = runs[m]
  if sweep > 1 && Tc
    (Tcmin-1E-10 <= temp <= Tcmax+1E-10) ? 0. : continue
  end
  sweep > 1 ? counterloop2 += 1 : 0
  tgpu = 0. #time spent computing on GPU
  t1 = time() #times the outerloop

  #initializing average values
  JavgE,JavgEsq,JavgM,JavgMsq,JavgMfour,JavgAbsM,JstdE,JstdM,JstdAbsM,sumsaveCount = 0.,0.,0.,0.,0.,0.,0.,0.,0.,0.
  singleHalf = -Float32(temp*0.5) #stores -T/2 for future reference
  function GPUcall()
    randpos = rand(Int32(1):Int32(latsize),Nloop*GPUloops) #define random positions
    randNum = log.(rand(Float32,Nloop*GPUloops))*singleHalf #define random numbers
    randpos_buff = cl.Buffer(Int32,ctx, (:r, :copy), hostbuf=randpos) #buffer data to GPU
    randNum_buff = cl.Buffer(Float32,ctx, (:r, :copy), hostbuf=randNum)

    res_buff = cl.Buffer(Float32, ctx, :w, GPUloops*7) #"magic number" of 7 here is for the number of observables wanted (see OpenCL part)
    p = cl.Program(ctx, source=MCloop_kernel) |> cl.build! #generates opencl kernel
    k = cl.Kernel(p, "MCloop")
    t3 = time()
    queue(k, GPUloops, nothing, psi_buff, intvec_buff, randpos_buff, randNum_buff, res_buff) #actual run of the GPU
    t4 = time()
    tgpu += t4-t3
    return res_buff
  end
  sumsaveCount = zeros(GPUloops)
  #thermalization
  mult = sweep == 1 ? delayloops : relaxloops
  for pqr = 1:mult
    result_buff = GPUcall()
  end

  #sets all states to a state near the observable
  if sweep == 1
    if qq == -nruns && !ordered
      s = deepcopy(cl.read(queue, result_buff)) #transfer resulting data from the GPU to the CPU
      s = reshape(s,GPUloops,7) #reshapes from a vector into a GPUloops x 7 matrix
      findval = sum(s[:,var])/GPUloops
      foundval = sortperm([abs(s[b,var]-findval) for b = 1:GPUloops])

      psi = deepcopy(cl.read(queue, psi_buff))
      psi = reshape(psi,GPUloops,latsize)
      copypsi = deepcopy(psi[foundval[1],:])
      for w = 1:GPUloops
        psi[w,:] = deepcopy(copypsi)
      end
      psi_buff = cl.Buffer(Int8, ctx, (:rw, :copy), hostbuf=psi)
    end
  else
    psi_buff = cl.Buffer(Int8, ctx, (:rw, :copy), hostbuf=reshape(geneticpsi[m],GPUloops*latsize))
  end

  #actual sweeps
  mult = sweep == 1 ? delaymult : maxmult
  for pqr = 1:mult
    result_buff = GPUcall()
    s = deepcopy(cl.read(queue, result_buff)) #transfer resulting data from the GPU to the CPU
    s = reshape(s,GPUloops,7) #reshapes from a vector into a GPUloops x 7 matrix

    sumsaveCount += Float64[s[w,7]==0 ? Float64(1) : Float64(s[w,7]) for w = 1:GPUloops] #saves the number of counts from the GPU

    JavgE += s[:,1] #adds averages to the stored average values on the CPU
    JavgEsq += s[:,2]
    JavgM += s[:,3]
    JavgMsq += s[:,4]
    JavgAbsM += s[:,5]
    JavgMfour += s[:,6]
  end

  #stores values for each thread
  threadE,threadEsq,threadM,threadMsq,threadAbsM,threadMfour = Float64[],Float64[],Float64[],Float64[],Float64[],Float64[],Float64[]
  threadCv,threadChiM,threadB = Float64[],Float64[],Float64[]
  floatconvert(fcount::Float64,x::Array{Float32,1}) = Float64[Float64(x[p])/fcount for p = 1:size(x,1)]
  for b = 1:GPUloops
    fcount = Float64(sumsaveCount[b])#floating point count
    En,Esq,avgM,Msq,AbsM,Mfour = floatconvert(fcount,Float32[JavgE[b],JavgEsq[b],JavgM[b],JavgMsq[b],JavgAbsM[b],JavgMfour[b]])
    push!(threadE,En) #pushed value to vector
    push!(threadEsq,Esq)
    push!(threadM,avgM) #didn't need average of M below, so only the vector is taken
    push!(threadMsq,Msq)
    push!(threadAbsM,AbsM)
    push!(threadMfour,Mfour)
    push!(threadCv,(Esq-En^2)/temp^2)#calculate specific heat for each thread
    push!(threadChiM,(Msq-AbsM^2)/temp)#calculate magnetic susceptibility for each thread
    push!(threadB,1-(Mfour/Msq^2)/3)#calculate binder cumulant for each thread
  end

  #calculate averages of values between all threads
  averages = avg(threadE,threadM,threadAbsM,threadMfour,threadEsq,threadMsq,threadCv,threadChiM,threadB)
  JavgE,JavgM,JavgAbsM,JavgMfour,JavgEsq,JavgMsq,specificHeat,susceptibility,binder = deepcopy(averages)
  #calculates errors for a given thread (standard deviation of the mean)
  errors = stDevsq(averages,threadE,threadM,threadAbsM,threadMfour,threadEsq,threadMsq,threadCv,threadChiM,threadB)
  JstdE,JstdM,JstdAbsM,JstdMfour,JstdEsq,JstdMsq,errorCv,errorChiM,errorB = deepcopy(errors)

  #loads select states into geneticpsi for the final sweep
  if sweep == 1
    gcounter = 1
    if (qq < 0)
      savedpsi[m] = deepcopy(reshape(cl.read(queue, psi_buff),GPUloops,latsize))
      savedE[m] = deepcopy(threadE)
    else
        #generate a random set of wavefunctions inside the hysteresis loop
        minval = min(databack[m,var+1]-sqrt(databack[m,var+2]*GPUloops),datafore[m,var+1]-sqrt(datafore[m,var+2]*GPUloops))
        maxval = max(databack[m,var+1]+sqrt(databack[m,var+2]*GPUloops),datafore[m,var+1]+sqrt(datafore[m,var+2]*GPUloops))
        psi = deepcopy(cl.read(queue, psi_buff))
        psi = reshape(psi,GPUloops,latsize)
        while true
          w = rand(1:GPUloops)
          randnumber = round(rand()) == 1.0
          if randnumber
            if ((minval <= savedE[m][w]/latsize <= maxval) | (maxval <= savedE[m][w]/latsize <= minval))
              geneticpsi[m][gcounter,:] = deepcopy(savedpsi[m][w,:])
              gcounter += 1
            end
          else
            if ((minval <= threadE[w]/latsize <= maxval) | (maxval <= threadE[w]/latsize <= minval))
              geneticpsi[m][gcounter,:] = deepcopy(psi[w,:])
              gcounter += 1
            end
          end
          gcounter > GPUloops ? break : 0
        end
    end
  end

  valtosave = [JavgE,JavgM,JavgAbsM,specificHeat,susceptibility,binder]
  errtosave = [JstdE,JstdM,JstdAbsM,errorCv,errorChiM,errorB]
  #record data
  t2 = time()
  if sweep == 1
    qq < 0 ? savedata(datafore,m,t2,t1,tgpu,valtosave,errtosave) : savedata(databack,m,t2,t1,tgpu,valtosave,errtosave)
  else
    savedata(data,counterloop2,t2,t1,tgpu,valtosave,errtosave)
  end
end #end sweep loop

end #end temperature scan loop
#write data to files
function savedata(name::String,dataArray::Array{Float64,2})
  f = open(name,"w")
  write(f,"#T        <E>      +/-E       <M>      +/-M      <|M|>     +/-|M|    C_v      +/-C_v    chi_|M|    +/-chi     B       +/-B\n")
  for i = 1:size(dataArray,1)
    write(f,join([(@sprintf "%.4f" dataArray[i,1]),"  "]))
    for k = 2:2:size(dataArray,2)-1
      write(f,join([(@sprintf "%.6f" dataArray[i,k]),"  ",(@sprintf "%.6f" sqrt(dataArray[i,k+1]))," "]))
    end
    write(f,"\n")
  end
  close(f)
end

savedata("dataforward_L=$(L).txt",datafore)
savedata("databackward_L=$(L).txt",databack)
savedata("data_L=$(L).txt",data)
