package recsys.eval;

import java.util.List;

import recsys.data.Rating;

public class RecSysEval {

	public static void rating(List<Rating> ratings){
		float mae = 0;
        float rmse = 0;
        int count = 0;
        for (Rating r : ratings)
        {
        	float error = r.rating - r.prediction;
            count++;
            mae += Math.abs(error);
            rmse += Math.pow(error, 2);
        }
        mae /= count;
        rmse /= count;
        rmse = (float)Math.sqrt(rmse);
        System.out.println("MAE: "+mae+" RMSE: "+rmse);
	}
	
}
