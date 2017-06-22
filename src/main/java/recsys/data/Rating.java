package recsys.data;

import nn4j.data.Data;

public class Rating extends Data{

	public int userID;
	public int itemID;
	public float rating;
	public long timestamp;
	public Rating(int userID, int itemID, float rating, long timestamp) {
		super();
		this.userID = userID;
		this.itemID = itemID;
		this.rating = rating;
		this.timestamp = timestamp;
	}
	
	public float prediction;
}
