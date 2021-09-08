#!/bin/tcsh


set Method = MHPsPRs

foreach case_type ( confirm death )
	foreach alpha_shape( 21 ) 
		foreach beta_scale( 5 7 )
			echo "python MHPsPRs.py --case_type ${case_type} --alpha_shape ${alpha_shape} --beta_scale ${beta_scale} >/dev/null" > ./job/job_case_${case_type}_alphashape_${alpha_shape}_betascale_${beta_scale}.job
		end
	end
	#
	foreach alpha_shape( 7 14 ) 
		foreach beta_scale( 5 )
			echo "python MHPsPRs.py --case_type ${case_type} --alpha_shape ${alpha_shape} --beta_scale ${beta_scale} >/dev/null" > ./job/job_case_${case_type}_alphashape_${alpha_shape}_betascale_${beta_scale}.job
		end
	end
end




