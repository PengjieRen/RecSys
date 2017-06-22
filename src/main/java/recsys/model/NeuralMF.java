package recsys.model;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.impl.UniformDistribution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import nn4j.cg.ComputationGraph;
import nn4j.cg.Dense;
import nn4j.cg.LookupTable;
import nn4j.data.Batch;
import nn4j.data.Data;
import nn4j.expr.Add;
import nn4j.expr.Concat;
import nn4j.expr.DefaultParamInitializer;
import nn4j.expr.Dropout;
import nn4j.expr.Expr;
import nn4j.expr.Parameter;
import nn4j.expr.Parameter.RegType;
import nn4j.expr.ParameterManager;
import nn4j.expr.ParameterManager.Updater;
import nn4j.expr.WeightInit;
import nn4j.loss.Loss;
import recsys.Constant;
import recsys.data.DataUtils;
import recsys.data.Rating;
import recsys.data.RecSysRatingDataLoader;
import recsys.eval.RecSysEval;

public class NeuralMF extends ComputationGraph{

	private LookupTable embTable;
	private LookupTable biasTable;
	private Parameter w1;
	private Parameter w2;

	public NeuralMF(ParameterManager pm,LookupTable embTable,LookupTable biasTable) {
		super(pm);
		this.embTable=embTable;
		this.biasTable=biasTable;
	}

	@Override
	public void parameters() {
		w1 = pm.createParameter(new DefaultParamInitializer(WeightInit.DISTRIBUTION,new UniformDistribution(-1f/20, 1f/20)).init(new int[] { 21, 10 }),RegType.L2,0.1f, true,false);
		w2 = pm.createParameter(new DefaultParamInitializer(WeightInit.DISTRIBUTION,new UniformDistribution(-1f/10, 1f/10)).init(new int[] { 11, 1 }),RegType.L2,0.1f, true,false);
	}

	@Override
	public Loss model(Batch batch, boolean training) {
		Expr user=batch.batchInputs[0][0];
		Expr item=batch.batchInputs[1][0];
		Expr userb=batch.batchInputs[2][0];
		Expr itemb=batch.batchInputs[3][0];
		
//		Expr dot=new InnerProduct(user, item);
//		Expr add=new Add(dot,userb,itemb);
//		Loss loss=new Loss(add,LossFunction.MSE);

		Expr concat=new Concat(user,item);
		Expr layer1=new Dense(concat, w1, Activation.RELU, true, training);
		Expr layer2=new Dense(layer1, w2, Activation.IDENTITY, true, training);
		Expr add=new Add(layer2,userb,itemb);
		Loss loss=new Loss(add,LossFunction.MSE);
		
		return loss;
	}

	@Override
	public void test(String run, List<Data> testData, File gt) {
		List<Rating> ratings=new ArrayList<Rating>();
		for(Data d : testData){
			ratings.add((Rating)d);
		}
		Batch batch=DataUtils.toBatch(embTable,biasTable, ratings);
		Loss model= model(batch,false);
		INDArray output=model.forward();
		for(int i=0;i<ratings.size();i++){
			ratings.get(i).prediction=output.getFloat(i);
		}
		
		RecSysEval.rating(ratings);
	}

	
	public static void main(String[] args) {
		DataTypeUtil.setDTypeForContext(DataBuffer.Type.FLOAT);
		System.setProperty("ndarray.order","c");
		Nd4j.getMemoryManager().togglePeriodicGc(false);
		
		int batchSize=100;
		ParameterManager pm = new ParameterManager(Updater.RMSPROP);
		RecSysRatingDataLoader all=new RecSysRatingDataLoader(pm, null,null, new File(Constant.root,"data/ml-100k/u.data") , batchSize);
		List<Data> data=all.data();
		List<String> names=new ArrayList<String>();
		for(Data d:data){
			Rating r=(Rating)d;
			if(!names.contains("U"+r.userID))
			{
				names.add("U"+r.userID);
			}
			if(!names.contains("I"+r.itemID))
			{
				names.add("I"+r.itemID);
			}
		}
		LookupTable embTable=new LookupTable(pm,names.size(),10,RegType.None,0.1f,true);
		embTable.init(names);
		
		names=new ArrayList<String>();
		for(Data d:data){
			Rating r=(Rating)d;
			if(!names.contains("Ub"+r.userID))
			{
				names.add("Ub"+r.userID);
			}
			if(!names.contains("Ib"+r.itemID))
			{
				names.add("Ib"+r.itemID);
			}
		}
		LookupTable biasTable=new LookupTable(pm,names.size(),1,RegType.L2,0.01f,true);
		biasTable.init(names);

		RecSysRatingDataLoader train=new RecSysRatingDataLoader(pm, embTable,biasTable, new File(Constant.root,"data/ml-100k/u2.base"), batchSize);
		RecSysRatingDataLoader test=new RecSysRatingDataLoader(pm, embTable,biasTable, new File(Constant.root,"data/ml-100k/u2.test"), batchSize);

		NeuralMF nmf=new NeuralMF(pm, embTable,biasTable);
		nmf.train(train, null, null, test.data(), null, 100);
	}
}
