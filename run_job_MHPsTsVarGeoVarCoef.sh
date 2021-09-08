#!/bin/tcsh
set Method=MHPsTsVarGeoVarCoef
foreach time ( `seq 1 $1` )
	perl ./Drone.pl ./job_${Method}/ ./lock_${Method}/ > ./error.text &
end
