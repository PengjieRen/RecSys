package recsys.data;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import nn4j.cg.LookupTable;
import nn4j.data.Batch;
import nn4j.data.Data;
import nn4j.data.DataLoader;
import nn4j.expr.ParameterManager;

public class RecSysRatingDataLoader extends DataLoader{

	private List<Data> data;
	private int pointer;
	private int batchSize;
	private LookupTable embTable;
	private LookupTable biasTable;
	
	public RecSysRatingDataLoader(ParameterManager pm,LookupTable embTable,LookupTable biasTable,File file,int batchSize) {
		super(pm);
		this.batchSize=batchSize;
		this.embTable=embTable;
		this.biasTable=biasTable;
		try{
			data=new ArrayList<Data>();
			BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
			while(br.ready()){
				String line=br.readLine();
				if(line.trim().length()>0){
					String[] temp=line.split("\t");
					Rating r=new Rating(Integer.parseInt(temp[0]),Integer.parseInt(temp[1]),Float.parseFloat(temp[2]),Long.parseLong(temp[3]));
					data.add(r);
				}
			}
			br.close();
			
		}catch(Exception e){
			e.printStackTrace();
		}
	}
	private Random rng=new Random();
	 protected void shuffle(List<Data> data){
			for(int i=0;i<data.size();i++){
				int index=rng.nextInt(data.size());
				Data p1=data.get(i);
				Data p2=data.get(index);
				data.set(index, p1);
				data.set(i, p2);
			}
		}

	@Override
	public Batch next() {
		int thisBatchSize=pointer+batchSize<=data.size()?batchSize:data.size()-pointer;
		
		List<Rating> ratings=new ArrayList<Rating>(); 
		for(int i=0;i<thisBatchSize;i++){
			ratings.add((Rating)data.get(pointer+i));
		}
		pointer+=thisBatchSize;
		return DataUtils.toBatch(embTable,biasTable,ratings);
	}

	@Override
	public boolean hasNext() {
		return pointer<data.size();
	}

	@Override
	public void reset() {
		pointer=0;
		shuffle(data);
	}

	@Override
	public List<Data> data() {
		return data;
	}

}
