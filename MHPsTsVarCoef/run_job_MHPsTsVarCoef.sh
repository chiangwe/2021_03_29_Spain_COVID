#!/bin/tcsh
set Method=MHPsTsVarGeoVarCoef
foreach time ( `seq 1 $1` )
	perl ./Drone.pl ./job/ ./lock/ > ./error.text &
end

