#!/bin/tcsh


set Method = MHPsTsVarCoef

foreach case_type ( confirm death )
	foreach bw ( 5 10 )
		foreach alpha_shape( 0 ) 
			foreach beta_scale( 0 )
				echo "python ${Method}.py --case_type ${case_type} --bw ${bw} --alpha_shape ${alpha_shape} --beta_scale ${beta_scale} >/dev/null" > ./job/job_case_${case_type}_bw_${bw}_alphashape_${alpha_shape}_betascale_${beta_scale}.job
			end
		end
	end
end




