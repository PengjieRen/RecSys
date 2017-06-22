package recsys.data;

import java.util.List;

import org.nd4j.linalg.factory.Nd4j;

import nn4j.cg.LookupTable;
import nn4j.data.Batch;
import nn4j.expr.Expr;

public class DataUtils {
	public static Batch toBatch(LookupTable embTable,LookupTable biasTable,List<Rating> ratings){
		Batch ret=new Batch();
		ret.batchInputs=new Expr[4][1];
		ret.batchGroundtruth=Nd4j.create(ratings.size(), 1);

		String[] users=new String[ratings.size()];
		String[] items=new String[ratings.size()];
		String[] usersb=new String[ratings.size()];
		String[] itemsb=new String[ratings.size()];
		for(int i=0;i<ratings.size();i++){
			users[i]="U"+ratings.get(i).userID;
			items[i]="I"+ratings.get(i).itemID;
			usersb[i]="Ub"+ratings.get(i).userID;
			itemsb[i]="Ib"+ratings.get(i).itemID;
			ret.batchGroundtruth.putScalar(i, ratings.get(i).rating);
		}
		ret.batchInputs[0][0]=embTable.get(users);
		ret.batchInputs[1][0]=embTable.get(items);
		ret.batchInputs[2][0]=biasTable.get(usersb);
		ret.batchInputs[3][0]=biasTable.get(itemsb);
		
		return ret;
	}
}
