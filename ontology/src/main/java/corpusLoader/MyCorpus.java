package corpusLoader;

import java.io.File;
import java.io.FileInputStream;   
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.ObjectOutputStream;
import java.io.Reader;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Random;

import org.jfree.data.json.impl.JSONObject;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonStreamParser;

import edu.eur.absa.Framework;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import net.arnx.jsonic.JSONException;




//change so that you can read in hashmap with noun/verb/etc DONE
public class MyCorpus{

	private String filelocation_review;
	private String filelocation_business;
	private List<String> restaurants = new ArrayList<String>();
	private Map<String, String> allTerms = new HashMap<String, String>();
	private String training_data_review;
	private String contrasting_data_review;
	
	public MyCorpus(String filelocation_review, String filelocation_business) {

		this.filelocation_review = filelocation_review;
		this.filelocation_business = filelocation_business;
		
	}
	
	public List<String> business_identifier() throws FileNotFoundException, UnsupportedEncodingException {
	   InputStream is_b = new FileInputStream(filelocation_business);
	   Reader r_b = new InputStreamReader(is_b, "UTF-8");
	   Gson gson_b = new GsonBuilder().create();
	   JsonStreamParser p = new JsonStreamParser(r_b);
	   while (p.hasNext()) {
	      JsonElement e = p.next();
	      if (e.isJsonObject()) {
	          business_identifier identifier = gson_b.fromJson(e, business_identifier.class);     
	          boolean isRestaurant = identifier.contains_key("RestaurantsPriceRange2");
	          if (isRestaurant == true) {
	        	  restaurants.add(identifier.get_id());
	          } 	          
	      }
	   }
	   return restaurants;

	}
		
	
	public void review_loader() throws FileNotFoundException, UnsupportedEncodingException {
		int counter = 0;
	    String pattern ="[\\p{Punct}&&[^@',&]]";
		Properties props = new Properties();
	    // set the list of annotators to run
	    props.setProperty("annotators", "tokenize,ssplit,pos,lemma");
	    // set a property for an annotator, in this case the coref annotator is being
	    // build pipeline
	    StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
		//MaxentTagger tagger = new MaxentTagger(filelocation_pos);
		InputStream is_r = new FileInputStream(filelocation_review);
		Reader r_r = new InputStreamReader(is_r, "UTF-8");
		Gson gson_r = new GsonBuilder().create();
		JsonStreamParser p = new JsonStreamParser(r_r);

		while (p.hasNext() && counter < 100000) {
			counter++;
			JsonElement e = p.next();
			if (e.isJsonObject()) {
				review review = gson_r.fromJson(e, review.class);
				if (restaurants.contains(review.get_id())) {
					if (review.get_text().length() < 500) {
						counter++; 
						Map<String, String> review_terms = review.stanford_pipeline_tagger(pipeline, pattern);
						allTerms.putAll(review_terms);
						System.out.println("size:" + allTerms.size() + "reviews processed: " + counter);	
					}			
				}
			}
		}
	}
	
	/**
	 * Creates domain specific data from full data set
	 * @throws FileNotFoundException
	 * @throws UnsupportedEncodingException
	 */
	public void getDomainTrainingData() throws FileNotFoundException, UnsupportedEncodingException {
		int counterRest = 0;
		int counterContr = 0;
		int counterProc = 0;
		training_data_review = "";
	    String pattern ="[\\p{Punct}&&[^@',&]]";
		Properties props = new Properties();
	    // set the list of annotators to run
	    props.setProperty("annotators", "tokenize,ssplit,pos,lemma");
	    // set a property for an annotator, in this case the coref annotator is being
	    // build pipeline
	    //StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
		//MaxentTagger tagger = new MaxentTagger(filelocation_pos);
		InputStream is_r = new FileInputStream(filelocation_review);
		Reader r_r = new InputStreamReader(is_r, "UTF-8");
		Gson gson_r = new GsonBuilder().create();
		JsonStreamParser p = new JsonStreamParser(r_r);
		long startTime = System.currentTimeMillis();
		int fileCountContr = 1 ;
		int fileCountRest = 1 ;
		int sentiment;
		boolean contrast = false;
		boolean restaurant = false;
		boolean training = false; // True for (post)training template, false for finetuning template
		boolean t5 = true; // True for T5 finetuning template, false for Roberta and BERT finetuning
		int no_reviews = 50000;
		String output_filename = "T5FineTune.txt";
		String reviewtext = "";
		while (p.hasNext()) {
			JsonElement e = p.next();
			if (e.isJsonObject()) {
				review review = gson_r.fromJson(e, review.class);
				if (restaurants.contains(review.get_id())) {
					if (review.get_text().length() < 500) {
						if (training) {
							reviewtext = review.get_text();
							reviewtext = reviewtext.replaceAll("[\\n\\t\\r ]", " ");
							training_data_review = training_data_review + ",|," + "\n" + reviewtext;
							counterRest++;
						} else if (t5) {
							reviewtext = review.get_text();
							reviewtext = reviewtext.replaceAll("[\\n\\t\\r ]", " ");
							if (review.get_stars() < 3) {
								training_data_review = training_data_review + "\n0\n" + reviewtext;
								counterRest++;
							} else if (review.get_stars() > 3) {
								training_data_review = training_data_review + "\n1\n" + reviewtext;
								counterRest++;
							}
						} else {
							reviewtext = review.get_text();
							reviewtext = reviewtext.replaceAll("[\\n\\t\\r ]", " ");
							if (review.get_stars() < 3) {
								training_data_review = training_data_review + ",|," + "\n" + "0,/," + reviewtext;
								counterRest++;
							} else if (review.get_stars() > 3) {
								training_data_review = training_data_review + ",|," + "\n" + "1,/," + reviewtext;
								counterRest++;
							}
							
						}
					}
				}	
				if (counterRest > no_reviews) {
					domain_file(output_filename);
					break;
				}
			}
		}
		System.out.println("Restaurant reviews :" + counterRest);
		System.out.println("total:" + counterProc);
		long elapsedTime = System.currentTimeMillis() - startTime;
		long elapsedSeconds = elapsedTime / 1000;
		long MinutesDisplay = elapsedSeconds / 60;
	}
	
	private char[] restaurant(int i, int j) {
		// TODO Auto-generated method stub
		return null;
	}
	
	
	/**
	 * Writes file containing domain specific data for training of BERT model
	 */
	public void domain_file(String fileName) {
		System.out.println("saving file..");
		JsonObject jsonObject = new JsonObject();
	    try {
	        File fileOne=new File(fileName);
	        FileOutputStream fos=new FileOutputStream(fileOne);
	        ObjectOutputStream oos=new ObjectOutputStream(fos);

	        oos.writeObject(training_data_review.toString());
	        oos.flush();
	        oos.close();
	        fos.close();
	    } catch(Exception e) {}
	}
	public void contrasting_file(String fileName) {
		System.out.println("saving file..");
		JsonObject jsonObject = new JsonObject();
	    try {
	        File fileOne=new File(fileName);
	        FileOutputStream fos=new FileOutputStream(fileOne);
	        ObjectOutputStream oos=new ObjectOutputStream(fos);

	        oos.writeObject(contrasting_data_review.toString());
	        oos.flush();
	        oos.close();
	        fos.close();
	    } catch(Exception e) {}
	}
	
		/*
		 * public void review_loader_own() throws FileNotFoundException,
		 * UnsupportedEncodingException { String[] needed_tags = {"VB","VBD", "VBG",
		 * "VBN","VBP","VBZ","VH","VHD","VHG","VHN","VHP","VHZ","VV","VVD","VVG","VVN",
		 * "VVP","VVZ","JJ","JJR","JJS","NN","NNS","NP","NPS","RB","RBR","RBS"}; ArrayList<String>
		 * needed_tags_l = new ArrayList<String>(Arrays.asList(needed_tags)); int
		 * counter = 0; MaxentTagger tagger = new MaxentTagger(filelocation_pos);
		 * InputStream is_r = new FileInputStream(filelocation_review); Reader r_r = new
		 * InputStreamReader(is_r, "UTF-8"); Gson gson_r = new GsonBuilder().create();
		 * JsonStreamParser p = new JsonStreamParser(r_r); while (p.hasNext()) { counter
		 * += 1; JsonElement e = p.next(); if (e.isJsonObject()) { Review review =
		 * gson_r.fromJson(e, Review.class); if (restaurants.contains(review.get_id()))
		 * { review.get_pos_tags(tagger); HashSet<String> terms_in_review =
		 * review.get_adj_noun_verb_new(needed_tags_l);
		 * allTerms.addAll(terms_in_review); System.out.println("size:" +
		 * allTerms.size() + "reviews processed: " + counter); //
		 * System.out.println(verbs); } }
		 * 
		 * } }
		 */

	
	public void write_to_file() {
		System.out.println("saving file..");
	    try {
	    	File fileOne=new File( Framework.OUTPUT_PATH + "Output_stanford_hashmap");
	        FileOutputStream fos=new FileOutputStream(fileOne);
	        ObjectOutputStream oos=new ObjectOutputStream(fos);

	        oos.writeObject(allTerms);
	        oos.flush();
	        oos.close();
	        fos.close();
	    } catch(Exception e) {}
	}
		
	
	public static void main(String args[]) throws IOException {
		// WHEN YOU RUN THE FILE you need to add review.json and business.json to the external data directory!
		Framework framework = new Framework();
		MyCorpus yelp_dataset = new MyCorpus(Framework.EXTERNALDATA_PATH + "review.json", Framework.EXTERNALDATA_PATH + "business.json");
		List<String> restaurants = yelp_dataset.business_identifier();
		yelp_dataset.review_loader();
		//yelp_dataset.write_to_file();
		yelp_dataset.getDomainTrainingData();

		}
	
}