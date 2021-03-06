package sentimentBuilder;

import java.io.BufferedReader; 
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.TreeMap;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.jena.tdb.sys.SystemTDB;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.core.*;

import edu.eur.absa.Framework;
import it.unimi.dsi.fastutil.doubles.DoubleList;
import it.unimi.dsi.fastutil.doubles.DoubleLists;

public class SentimentWordProcessor {
	private Map<String, double[]> word_vec_PT = new HashMap<String, double[]>();
	private Map<String, Integer> sentiment_mentions = new HashMap<String, Integer>();

	private Map<String, Map<double[], double[]>> finalMap = new HashMap<>(); 

	private Map<String, double[]> word_vec_FT = new HashMap<String, double[]>();
	private Map<String, Map<String,String>> clustered_sentiment = new HashMap<String,Map<String,String>>();


	public SentimentWordProcessor(String fileloc_yelp, String fileloc_sent) throws ClassNotFoundException, IOException{
		read_file("finalMap", fileloc_yelp);
		read_file("sentiment", fileloc_sent);		
		//read_word2vec_file();
		for(Map.Entry<String, Map<double[], double[]>> entry : finalMap.entrySet()) {
			Map.Entry<double[], double[]> entry2 = finalMap.get(entry.getKey()).entrySet().iterator().next(); 
			word_vec_FT.put(entry.getKey(), entry2.getValue());  // woord en FT vec toegevoegd. 
			word_vec_PT.put(entry.getKey(), entry2.getKey()); 
		}
	}
	/**
	public void read_word2vec_file() throws IOException {
		File test = new File(Framework.LARGEDATA_PATH + "refined_model.json");

		Map<String,List<Double>> result = new ObjectMapper().readValue(test, Map.class);

		for (Map.Entry<String, List<Double>> entry : result.entrySet()) 
		{
			String word = entry.getKey();
			List<Double> vector_prem = new ArrayList<Double>(entry.getValue());
			Double[] array = vector_prem.toArray(new Double[vector_prem.size()]);
			double[] word_vector = ArrayUtils.toPrimitive(array);
			word_vec_FT.put(word, word_vector);
		}
	}
	 **/

	public void read_file(String dataset, String filelocation) throws IOException, ClassNotFoundException {
		if (dataset == "finalMap") {
			File toRead_yelp=new File(filelocation);
			FileInputStream fis_yelp=new FileInputStream(toRead_yelp);
			ObjectInputStream ois_yelp =new ObjectInputStream(fis_yelp);
			finalMap =(HashMap<String, Map<double[], double[]>>)ois_yelp.readObject();
			ois_yelp.close();
			fis_yelp.close();	
		}
		if (dataset == "sentiment") {
			File toRead_sent=new File(filelocation);
			FileInputStream fis_sent=new FileInputStream(toRead_sent);
			ObjectInputStream ois_sent =new ObjectInputStream(fis_sent);
			sentiment_mentions =(HashMap<String,Integer>)ois_sent.readObject();
			ois_sent.close();
			fis_sent.close();	
		}
	}


	public double get_cosine_similarity(double[] vec1, double[] vec2) {
		return (dotProduct(vec1, vec2)/(getMagnitude(vec2) * getMagnitude(vec1)));
	}

	public double generate_sentiment_scores() {
		String[] negative_sentiment_seeds = {"bad", "awful", "horrible", "terrible", "poor", "lousy", "shitty", "horrid"};
		String[] positive_sentiment_seeds = {"good", "decent", "great", "tasty", "fantastic", "solid", "yummy", "terrific"};
		List<double[]> negSeedList = getSeedList(negative_sentiment_seeds);
		List<double[]> posSeedList = getSeedList(positive_sentiment_seeds); 

		for (Map.Entry<String, Integer> sentiment_word : sentiment_mentions.entrySet()) {
			double[] wordembedding = word_vec_FT.get(sentiment_word.getKey());
			double max_sim_pos = 0;
			double max_sim_neg = 0;

			for(double[] wordvec : negSeedList) {
				System.out.println(wordvec);
				if (wordvec != null && wordembedding != null) {
					double cosine_sim = get_cosine_similarity(wordvec, wordembedding);
					if (cosine_sim > max_sim_neg) {
						max_sim_neg = cosine_sim;
					}
				}
			}
			for (double[] wordvec : posSeedList) {
				double cosine_sim = get_cosine_similarity(wordvec, wordembedding);
				if (cosine_sim > max_sim_pos) {
					max_sim_pos = cosine_sim;
				}
			}
			if (max_sim_pos>max_sim_neg) {
				System.out.println(sentiment_word.getKey() + " is a positive word, pos similarity: " + max_sim_pos + " neg similarity: " + max_sim_neg);
			}
			else {
				System.out.println(sentiment_word.getKey() + " is a negative word, pos similarity: " + max_sim_pos + " neg similarity: " + max_sim_neg);

			}
		}

		return 0;
	}

	public List<double[]> getSeedList(String[] seeds){
		List<double[]> seedList = new ArrayList<>(); 
		for(String seed : seeds) {
			if(word_vec_FT.containsKey(seed)) {
				seedList.add(word_vec_FT.get(seed));
			}
			else {
				for(Map.Entry<String, double[]> entry : word_vec_FT.entrySet()) {
					if(entry.getKey().contains(seed) && entry.getKey().contains("#")) {
						seedList.add(entry.getValue());
					}
				}
			}

		}
		return seedList; 
	}


	public String generate_sentiment_score(String sentiment_word) {
		String[] negative_sentiment_seeds = {"bad", "awful", "horrible", "terrible", "poor", "lousy", "shitty", "horrid"};
		String[] positive_sentiment_seeds = {"good", "decent", "great", "tasty", "fantastic", "solid", "yummy", "terrific"};
		List<double[]> negSeedList = getSeedList(negative_sentiment_seeds);
		List<double[]> posSeedList = getSeedList(positive_sentiment_seeds); 

		double[] wordembedding = word_vec_FT.get(sentiment_word);
		double max_sim_pos = 0;
		double max_sim_neg = 0;

		for(double[] wordvec : negSeedList) {
			System.out.println(wordvec);
			if (wordvec != null && wordembedding != null) {
				double cosine_sim = get_cosine_similarity(wordvec, wordembedding);
				if (cosine_sim > max_sim_neg) {
					max_sim_neg = cosine_sim;
				}
			}	
		}
		for (double[] wordvec : posSeedList) {
			System.out.println(wordvec);
			if (wordvec != null && wordembedding != null) {
				double cosine_sim = get_cosine_similarity(wordvec, wordembedding);
				if (cosine_sim > max_sim_pos) {
					max_sim_pos = cosine_sim;
				}
			}
		}
		if (max_sim_pos>max_sim_neg) {
			return "positive";
		}
		else {
			return "negative";
		}
	}


	public void save_to_file(Map<String,double[]> word_vec, String filelocation) {
		System.out.println("saving file..");
		try {
			File fileOne=new File(filelocation);
			FileOutputStream fos=new FileOutputStream(fileOne);
			ObjectOutputStream oos=new ObjectOutputStream(fos);

			oos.writeObject(word_vec);
			oos.flush();
			oos.close();
			fos.close();
		} catch(Exception e) {}
	}


	public List<String> get_closeness_mentionclasses(String[] mention_words, String word) {

		Map<Double,String> ranked_similarities = new TreeMap<Double, String>(new DescOrder());
		Map<String, Double> invRankSim = new HashMap<String, Double>(); 
		List<String> foodDrinkSim = new ArrayList<String>(); 
		for (String mention: mention_words) {
			double cosine_similarity = get_cosine_similarity(word_vec_PT.get(mention), word_vec_PT.get(word)); 
			ranked_similarities.put(cosine_similarity, mention);
		}
		
		for(Map.Entry<Double, String> entry1 : ranked_similarities.entrySet()) {
			invRankSim.put(entry1.getValue(), entry1.getKey());
		}
		
		for(Map.Entry<Double, String> entry : ranked_similarities.entrySet()) {
			if(!foodDrinkSim.contains(entry.getValue())) {
				foodDrinkSim.add(entry.getValue());
				if(entry.getValue().equals("food")) {
					foodDrinkSim.add("drinks");
				}
				else if(entry.getValue().equals("drinks")) {
					foodDrinkSim.add("food");
				}
			}
		}
	
		//Collections.reverse(foodDrinkSim);
		System.out.println(foodDrinkSim);
	
		return foodDrinkSim;
	}

	public Map<String, Map<String,String>> create_sentiment_links() {
		Scanner scan = new Scanner(System.in);
		String[] mention_words = {"ambience", "drinks","food","service","price","location","quality", "style", "options", "experience", "restaurant"};

		for (Map.Entry<String, Integer> sentiment_word : sentiment_mentions.entrySet()) {


			String polarity = generate_sentiment_score(sentiment_word.getKey()); // returns positive or negative

			// now we get a ranked list of mentionclasses based on closeness, IF it is type not type 1

			if(sentiment_word.getValue()==1) // type 1 words
			{ 
				
				// type 1, no need to attach it to a specific class
				System.out.println(sentiment_word.getKey() +" "+ polarity + "type 1"); 

				boolean wrong_entry_2 = true;
				while (wrong_entry_2) {
					if (polarity.equals("positive")){
						
						System.out.println("Is {" + sentiment_word.getKey()+ "} a positive word? Please enter Yes (y) or No (n) ");
						String input_2 = scan.nextLine();
						if (input_2.equals("y")) {
							// ADD TO POSITIVE WORDS IN MAP MAP THING
							wrong_entry_2 = false;
							addToMap("generic", sentiment_word.getKey(), "positive");
							System.out.println(clustered_sentiment);
						}
						else if (input_2.equals("n")) {
							addToMap("generic", sentiment_word.getKey(), "negative");
							wrong_entry_2= false;
						}
						else {
							System.out.println("Please enter a valid character");
						}
					
						wrong_entry_2 = false; //Use code above for user input, use this code for automatic sentiment prediction
						addToMap("generic", sentiment_word.getKey(), "positive");
						System.out.println(clustered_sentiment);
					}
					else {
						
						System.out.println("Is {" + sentiment_word.getKey()+ "} a negative word? Please enter Yes (y) or No (n) ");
						String input_2 = scan.nextLine();
						if (input_2.equals("y")) {
							addToMap("generic", sentiment_word.getKey(), "negative");
							wrong_entry_2 = false;
						}
						else if (input_2.equals("n")) {
							addToMap("generic", sentiment_word.getKey(), "positive");
							wrong_entry_2= false;

						}
						else {
							System.out.println("Please enter a valid character");
						}
						
						addToMap("generic", sentiment_word.getKey(), "negative");//Use code above for user input, use this code for automatic sentiment prediction
						wrong_entry_2 = false;
					}
				}
			}


			// It is is type 2 or 3 words, we use closeness to mentionclasses to determine which contexts the word
			// belongs to
			else { 
				List<String> similarities = get_closeness_mentionclasses(mention_words, sentiment_word.getKey());
				go_past_mention_classes: {
					for(String mention_classes: similarities) {
						boolean wrong_entry = true;

						while(wrong_entry) {
							System.out.println("Does sentimentword {"+sentiment_word.getKey()+"} belong to mention_class {"+mention_classes+"}? Please enter Yes (y) or No (n)"); 
							String input = scan.nextLine();

							if (input.equals("y")) {
								wrong_entry = false;
								boolean wrong_entry_2 = true;

								while (wrong_entry_2) {
									if (polarity.equals("positive")){
										
										System.out.println("Is {" + sentiment_word.getKey()+ "} a positive word? Please enter Yes (y) or No (n) ");
										String input_2 = scan.nextLine();
										if (input_2.equals("y")) {
											// ADD TO POSITIVE WORDS IN MAP MAP THING
											wrong_entry_2 = false;
											addToMap(mention_classes, sentiment_word.getKey(), "positive");
											//System.out.println(clustered_sentiment);
										}
										else if (input_2.equals("n")) {
											addToMap(mention_classes, sentiment_word.getKey(), "negative");
											wrong_entry_2= false;

										}
										else {
											System.out.println("Please enter a valid character");
										}
										
										wrong_entry_2 = false;//Use code above for user input, use this code for automatic sentiment prediction
										addToMap(mention_classes, sentiment_word.getKey(), "positive");
									}
									else {
										
										System.out.println("Is {" + sentiment_word.getKey()+ "} a negative word? Please enter Yes (y) or No (n) ");
										String input_2 = scan.nextLine();
										if (input_2.equals("y")) {
											addToMap(mention_classes, sentiment_word.getKey(), "negative");
											wrong_entry_2 = false;
										}
										else if (input_2.equals("n")) {
											addToMap(mention_classes, sentiment_word.getKey(), "positive");
											wrong_entry_2= false;

										}
										else {
											System.out.println("Please enter a valid character");
										}
										
										addToMap(mention_classes, sentiment_word.getKey(), "negative");
										wrong_entry_2 = false;//Use code above for user input, use this code for automatic sentiment prediction
									}
								}
							}

							else if (input.equals("n")) {
								wrong_entry=false;
								break go_past_mention_classes;
							}

							else {
								System.out.println("Please enter a valid character");
							}
						}
					}
				}
			}

		}
		return clustered_sentiment;
	}


	public void addToMap(String mapKey, String word_to_add, String polarity) {
		Map<String, String> itemsList = clustered_sentiment.get(mapKey);

		// if list does not exist create it
		if(itemsList == null)  {
			itemsList = new HashMap<String, String>(); 
			itemsList.put(word_to_add, polarity);
			clustered_sentiment.put(mapKey, itemsList); 
		}     
		else 
		{ // add if item is not already in list
			if(!itemsList.containsKey(word_to_add)) itemsList.put(word_to_add, polarity); 
		} 
	}


	public static void main(String args[]) throws IOException, ClassNotFoundException {
		SentimentWordProcessor sent_calc = new SentimentWordProcessor(Framework.OUTPUT_PATH + "finalMap", Framework.OUTPUT_PATH + "sentiment_mentions");
		//sent_calc.generate_sentiment_scores();
		sent_calc.create_sentiment_links();
		for(Map.Entry<String, Map<String, String>> entry : sent_calc.clustered_sentiment.entrySet()) {
			System.out.println(entry.getKey() + " " + entry.getValue());
		}
	}




	public static double dotProduct(double[] a, double[] b) {
		double sum = 0;
		for (int i = 0; i < a.length; i++) {
			sum += a[i] * b[i];
		}
		return sum;
	}
	public static double getMagnitude(double[] vec_1) {
		double magnitude = 0;
		for (int j = 0; j < vec_1.length; j++) {
			magnitude += Math.pow(vec_1[j],2);
		}
		return Math.sqrt(magnitude);
	}
	static class DescOrder implements Comparator<Double>{
		@Override
		public int compare(Double o1, Double o2) {
			return o2.compareTo(o1);
		}
	}
}