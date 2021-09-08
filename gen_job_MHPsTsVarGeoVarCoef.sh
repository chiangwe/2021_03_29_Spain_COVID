#!/bin/tcsh


set Method = MHPsTsVarGeoVarCoef

foreach bw ( 1 5 10 100 )
	foreach alpha_shape( 7 14  ) 
		foreach beta_scale( 5 )
			echo "python MHPsTsVarGeoVarCoef.py --bw ${bw} --alpha_shape ${alpha_shape} --beta_scale ${beta_scale} >/dev/null" > ./job_${Method}/job_bw_${bw}_alphashape_${alpha_shape}_betascale_${beta_scale}.job
		end
	end
end




