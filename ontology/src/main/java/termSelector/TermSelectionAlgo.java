package termSelector;

import java.awt.Frame;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File; 
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.io.Writer;
import java.math.RoundingMode;
import java.nio.file.Files;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Scanner;
import java.util.Arrays;


//import it.unimi.dsi.fastutil.Arrays;

import java.util.Set;
import java.util.TreeMap;
import java.util.regex.Pattern;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.languagetool.JLanguageTool;
import org.languagetool.Language;
import org.languagetool.language.AmericanEnglish;
import org.languagetool.rules.RuleMatch;

import com.apporiented.algorithm.clustering.AverageLinkageStrategy;
import com.apporiented.algorithm.clustering.Cluster;
import com.apporiented.algorithm.clustering.ClusteringAlgorithm;
import com.apporiented.algorithm.clustering.DefaultClusteringAlgorithm;
import com.apporiented.algorithm.clustering.visualization.DendrogramFrame;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import akka.japi.Pair;
import ch.qos.logback.core.recovery.ResilientSyslogOutputStream;
import edu.eur.absa.Framework;
import edu.smu.tspell.wordnet.Synset;
import edu.smu.tspell.wordnet.WordNetDatabase;

public class TermSelectionAlgo {
	public Map<String, List<double[]>> word_vec_PT = new HashMap<String, List<double[]>>();
	private Map<String, List<double[]>> word_vec_FT = new HashMap<String, List<double[]>>();
	private Map<String, String> allTerms = new HashMap<String, String>();
	private Map<String,double[]> wordVectorHighestTS = new HashMap<>();
	private Map<Double, String> term_scores_test = new TreeMap<>(new DescOrder());

	private Map<String,String> aspect_mentions = new HashMap<String, String>();
	private Map<String,Integer> sentiment_mentions = new HashMap<String, Integer>();
	private HashSet <String> allAccepted = new HashSet<>(); 

	private Map<double[], String> mention_classes_map = new HashMap<>();
	private List<Double> term_scores= new ArrayList<>();
	private Map<String, String> acceptedTerms = new HashMap<String, String>();

	private List<String> mention_words;
	private List<String> generic_words; 

	private Set<String> irrWords = new HashSet<>(); 

	public Map<String, Map<double[], double[]>> finalMap = new HashMap<>(); 

	private double max_score_noun;
	private double max_score_adj;
	private double max_score_verb;
	private double max_score_adv; 

	/**
	 * Constructor for TermSelectionAlgo, with contrasting google domain for term selection task
	 * @param filelocation_google
	 * @param filelocation_yelp
	 * @param filelocation_terms
	 * @throws ClassNotFoundException 
	 */
	public TermSelectionAlgo(String filelocation_PT, String filelocation_FT, String filelocation_terms) throws ClassNotFoundException, IOException {
		readIrrWords(Framework.EXTERNALDATA_PATH + "NLTK_stopwords");
		read_file("allTerms", filelocation_terms);
		
		/**
		System.out.println("reading PT");
		read_file("preTrained", filelocation_PT);
		System.out.println("reading FT..");
		read_file("FineTuned", filelocation_FT);
		System.out.println("creating final map..");
		finalMap = createFinalMap(word_vec_PT, word_vec_FT);
		save_to_file_map_in_map(finalMap, Framework.OUTPUT_PATH + "finalMap");
		//**/
	
		read_file("finalMap", Framework.OUTPUT_PATH + "finalMap"); 
		save_to_file_map_in_map(finalMap, Framework.OUTPUT_PATH + "finalMap");
		generic_words = Arrays.asList("good", "bad", "mediocre", "expensive", "hate", "great", "excellent", "enjoy", "liked", "perfectly", "well", "poorly", "badly", "tasty"); 
		get_vectors_mention_classes(generic_words, 0); 
		mention_words = Arrays.asList("ambience","drinks","food","service","price","location","quality", "style", "options", "experience","restaurant");
		get_vectors_mention_classes(mention_words, 1);

		createMentionClusters();
		
	}

	/**
	 * Constructor for TermSelectionAlgo, without google file or hashset since this is only needed for the synonyms method
	 * @param filelocation_yelp
	 * @param filelocation_terms
	 * @throws ClassNotFoundException
	 * @throws IOException
	 */
	public TermSelectionAlgo(String filelocation_yelp, String filelocation_terms) throws ClassNotFoundException, IOException {
		read_file("yelp", filelocation_yelp);
	}

	public Map<String, List<double[]>> getYelpMap(){
		return word_vec_PT; 
	}

	public void read_file(String dataset, String filelocation) throws IOException, ClassNotFoundException {
		if (dataset == "FineTuned") {
			word_vec_FT = read_file2(filelocation); 

		}
		if (dataset == "preTrained") {
			word_vec_PT = read_file2(filelocation); 
		}

		if (dataset == "allTerms") {
			File toRead_terms=new File(filelocation);
			FileInputStream fis_terms=new FileInputStream(toRead_terms);
			ObjectInputStream ois_terms =new ObjectInputStream(fis_terms);
			allTerms =(HashMap<String,String>)ois_terms.readObject();

			ois_terms.close();
			fis_terms.close();	
		}
		if(dataset == "finalMap") {
			File toRead_terms=new File(filelocation);
			FileInputStream fis_terms=new FileInputStream(toRead_terms);
			ObjectInputStream ois_terms =new ObjectInputStream(fis_terms);
			finalMap =(Map<String, Map<double[], double[]>>)ois_terms.readObject();

			ois_terms.close();
			fis_terms.close();
		}
	}
	/**
	 * Method that creates an averaged vector.
	 * @param stringList List of vector names
	 * @param term_wordvector Map with vectors and corresponding names
	 * @return 
	 */
	public static double[] getAverage(List<String> stringList, Map<String, double[]> term_wordvector){
		List<double[]> temp = new ArrayList<>();
		int length = 0; 
		for(Map.Entry<String, double[]> entry : term_wordvector.entrySet()) {
			length = entry.getValue().length; 
		}
		double[] toAverage = new double[length]; 
		for(String term : stringList) {
			if(!term.contains("#")) {
				double[] vector = term_wordvector.get(term); 
				temp.add(vector); 
			}
		}
		for(int i = 0; i < toAverage.length; i++) {
			for(double[] vec : temp) {
				toAverage[i] += vec[i]; 
			}
		}

		for(int i = 0; i < toAverage.length; i++) {
			toAverage[i] = toAverage[i] / temp.size();
		}
		return toAverage; 
	}

	public String fixHashtags(String[] hashtags) {
		StringBuffer myString = new StringBuffer(); 
		for(int i = 0; i < hashtags.length; i++) {
			hashtags[i] = hashtags[i].replaceAll("#", "");
			myString.append(hashtags[i]);
		}	
		return myString.toString();
	}

	public double[] fixVector(double[][] vector) {

		double[] tempVec = new double[vector[0].length];
		double[] myVector = new double[vector[0].length];
		for(int i = 0; i < vector[0].length; i++) {
			for(int j = 0; j < vector.length; j++) {
				tempVec[i] += vector[j][i]; 
			}
		}
		for(int i = 0; i < vector[0].length; i++) {
			myVector[i] = tempVec[i]/vector[0].length; 
		}
		return myVector; 
	}

	/**
	 * Method that reads the specific format for our BERT embeddings. It calls multiple methods like fixVector, fixHashtags, createKey and createVector. 
	 * @param filelocation
	 * @return
	 * @throws IOException
	 * @throws ClassNotFoundException
	 */
	public Map<String, List<double[]>> read_file2(String filelocation) throws IOException, ClassNotFoundException {
		String filePath = filelocation; 
		Map<String, List<double[]>> myMap = new HashMap<String, List<double[]>>(); 
		String line; 
		BufferedReader reader = new BufferedReader(new FileReader(filePath)); 
		while((line = reader.readLine()) != null)
		{
			line = line.substring(1, line.length() - 2);

			String[] parts = line.split("}, ");
			for(int i = 0; i < parts.length -1; i++) 
			{
				String key = createKey(parts[i]);
				if(!Pattern.matches("[\\p{Punct}\\p{IsPunctuation}]", key) && !key.contains("\\"))
				{
					double[] vector = createVector(parts[i]);  
					if(!parts[i].contains("#") && !parts[i+1].contains("#") &&  !myMap.containsKey(key)) 
					{
						List<double[]> myList = new ArrayList<>();
						myList.add(vector);
						myMap.put(key, myList); 
					}
					else if(!parts[i].contains("#") && !parts[i+1].contains("#") && myMap.containsKey(key)) 
					{
						myMap.get(key).add(vector); 
					}
					else if(!parts[i].contains("#") && parts[i+1].contains("#")) 
					{
						List<String> myString = new ArrayList<>();  
						myString.add(key);
						List<double[]> myVector = new ArrayList<>(); 
						myVector.add(vector); 
						while(parts[i+1].contains("#") && i < parts.length - 2)
						{
							myString.add(createKey(parts[i+1])); 
							myVector.add(createVector(parts[i+1])); 
							i++; 
						}
						double[][] myMatrix = new double[myVector.size()][vector.length];
						myMatrix = myVector.toArray(myMatrix);
						String[] myKeys = new String[myString.size()]; 
						for(int j = 0; j < myString.size(); j++) 
						{
							myKeys[j] = myString.get(j); 
						}
						if(!myMap.containsKey(fixHashtags(myKeys)))
						{
							List<double[]> myList = new ArrayList<>();
							myList.add(fixVector(myMatrix)); 
							myMap.put(fixHashtags(myKeys), myList);
						}
						else 
						{
							myMap.get(fixHashtags(myKeys)).add(fixVector(myMatrix)); 
						}
					}

				}
			}
			if(!parts[parts.length-1].contains("#") && parts[parts.length-2].contains("#")) {
				String key = createKey(parts[parts.length-1]);
				double[] vector = createVector(parts[parts.length-1]); 
				if(!myMap.containsKey(key)) {
					List<double[]> myList = new ArrayList<>();
					myList.add(vector);
					myMap.put(key, myList); 
				}
				else {
					myMap.get(key).add(vector); 
				} 
			}
		}
		reader.close();
		return myMap;	
	}

	public double[] createVector(String myString) {
		String[] toMap = myString.split(": "); 
		String myArray = toMap[3].substring(2, toMap[3].length() - 16); 
		String[] splitted = myArray.split(", "); 
		double[] myDouble = new double[splitted.length]; 
		for(int i = 0; i < splitted.length; i++) {
			myDouble[i] = Double.valueOf(splitted[i]); 
		}
		return myDouble; 
	}

	public String createKey(String myString) {
		String[] toMap = myString.split(":");
		String key = "";
		if(toMap[2].length() > 11) {
			key = toMap[2].substring(2, toMap[2].length() - 11); 
		}
		return key; 
	}

	public Map<String, Map<double[], double[]>> createFinalMap(Map<String, List<double[]>> preTrained, Map<String, List<double[]>> fineTuned){
		for(Map.Entry<String, List<double[]>> entry : preTrained.entrySet()) 
		{
			Map<double[], double[]> vectorMap = new HashMap<>();
			List<double[]> myList = entry.getValue();
			if (entry.getKey().length() > 0) {
				String word = entry.getKey();
				for(double[] vector : myList) 
				{
					int index = myList.indexOf(vector); 
					double[] vector2 = fineTuned.get(word).get(index); 
					vectorMap.put(vector, vector2);
				}
				finalMap.put(word, vectorMap); 
			}
		}
		finalMap = removeIrrelevantTerms(finalMap); 
		finalMap = averageVectors(finalMap); 
		return finalMap; 
	}

	public void get_vectors_mention_classes(List<String> words, int index) {
		for (String mention: words) 
		{			
			Map.Entry<String, Map<double[], double[]>> entry = finalMap.entrySet().iterator().next(); 
			Map<double[], double[]> entryMap = finalMap.get(mention); 
			if(entryMap != null) {
				Map.Entry<double[], double[]> entry2 = entryMap.entrySet().iterator().next(); 
				int length = entry2.getKey().length
						; 
				double[] vectorPT = new double[length];
				double[] vectorFT = new double[length];
				for(int i = 0; i < length; i++) 
				{
					for(double[] vec : entryMap.keySet()) 
					{
						vectorPT[i] += vec[i]; 
					}
					for(double[] vec : entryMap.values()) {
						vectorFT[i] += vec[i]; 
					}
				}
				for(int i = 0; i < length; i++) {
					vectorPT[i] = vectorPT[i] / entryMap.keySet().size();
					vectorFT[i] = vectorFT[i] / entryMap.keySet().size(); 
				} 
				if(index == 1) {
					mention_classes_map.put(vectorPT, mention);
				}
				entryMap.clear(); 
				entryMap.put(vectorPT, vectorFT);
			}
		}
	}


	//Checkt nu voor meerdere general vectors voor woord 'i' ipv 1 vector per woord.
	public double get_domain_similarity(List<double[]> general_list, double[] domain_vec) {	
		double max = -1; 
		for(double[] vector : general_list) {
			double answer = dotProduct(vector, domain_vec)/(getMagnitude(domain_vec) * getMagnitude(vector));
			if(answer > max) {
				max = answer; 
			}
		}
		return max; 
	}

	// Mention classes are: Ambience, Drinks, Food, Service, Price, Restaurant, Location, Quality, Style, Options, Experience
	public Map<Double, String> get_mention_class_similarity(double[] domain_vec) {
		double max = -1;
		String mention = ""; 
		Map<Double, String> myMap = new HashMap<>(); 
		for(Map.Entry<double[], String> entry : mention_classes_map.entrySet()) {
			double[] vector = entry.getKey(); 
			double similarity = (dotProduct(vector, domain_vec))/(getMagnitude(domain_vec) * getMagnitude(vector));
			if (similarity > max) {
				max = similarity;
				mention = entry.getValue(); 
			}
		}
		myMap.put(max, mention);
		return myMap;  
	}


	public Map<Double, String> get_term_score(double[] domain_vec) {

		return get_mention_class_similarity(domain_vec); 
		
	}
	/**
	 * This class makes sure that all terms with a MCS lower than a threshold are removed from finalMap. 
	 * It also clusters the vectors based on MCS and adds #MENTIONCLASS if a word has vectors with high MCS's at multiple mention classes.  
	 * 
	 */
	public void createMentionClusters(){ 
		double threshold = 0.68; 
		Map<String, Map<double[], double[]>> tempMap =  new HashMap<>(); 
		Iterator<Map.Entry<String, Map<double[], double[]>>> it = finalMap.entrySet().iterator(); 
		while(it.hasNext()) 
		{	
			Map.Entry<String, Map<double[], double[]>> entry = it.next(); 
			Map<String, List<double[]>> mentionMap = new HashMap<>(); 

			Map<double[], double[]> myMap = entry.getValue(); 

			for(double[] vector : myMap.keySet())  
			{	

				Map<Double, String> scoreMap = get_term_score(vector);
				Map.Entry<Double, String> entry2 = scoreMap.entrySet().iterator().next(); 

				String mention = entry2.getValue();
				double score = entry2.getKey(); 
				if(score > threshold) {
					if(!mentionMap.containsKey(mention)) {
						List<double[]> list = new ArrayList<>();

						list.add(vector); 
						mentionMap.put(mention, list);
					}
					else{
						mentionMap.get(mention).add(vector); 
					}
				}
			}
			if(!mentionMap.isEmpty()) {
				it.remove();
			}
			for(Map.Entry<String, List<double[]>> mentionEntry : mentionMap.entrySet()) {
				int length = mentionEntry.getValue().get(0).length; 
				double[] toAveragePT = new double[length]; 
				double[] toAverageFT = new double[length];
				for(int i = 0; i < length; i++) 
				{
					for(double[] vec : mentionEntry.getValue()) {
						toAveragePT[i] += vec[i];
						double[] vecFT = myMap.get(vec); 
						toAverageFT[i] += vecFT[i];	
					}	
				}
				for(int i = 0; i < toAverageFT.length; i++) 
				{
					toAveragePT[i] = toAveragePT[i] / mentionEntry.getValue().size();
					toAverageFT[i] = toAverageFT[i] / mentionEntry.getValue().size();
				} 
				Map<double[], double[]> newMap = new HashMap<>(); 
				newMap.put(toAveragePT, toAverageFT);
				String name; 
				if(mentionMap.size() > 1) {
					name = entry.getKey() + "#" + mentionEntry.getKey();
				}
				else {
					name = entry.getKey(); 
				} 
				tempMap.put(name, newMap); 
			}
		}
		for(Map.Entry<String, Map<double[], double[]>> entry : tempMap.entrySet()) {
			finalMap.put(entry.getKey(), entry.getValue()); 
		}

	}

	public void create_word_term_score() {
		System.out.println("creating word term score...");

		for (Map.Entry<String, Map<double[], double[]>> entry : finalMap.entrySet()) 
		{
			Map<double[], double[]> tempMap = entry.getValue();
			Map.Entry<double[], double[]> tempEntry = tempMap.entrySet().iterator().next(); 
			int length = tempEntry.getKey().length;
			double max = Integer.MIN_VALUE;  
			double[] myVector = new double[length];
			Map<double[], double[]> myMap = entry.getValue(); 
			for(double[] vector : myMap.keySet())  
			{	
				Map<Double, String> scoreMap = get_term_score(vector);
				Map.Entry<Double, String> entry2 = scoreMap.entrySet().iterator().next(); 
				double term_score = entry2.getKey();
				if(term_score > max) {
					max = term_score; 
					myVector = vector; 
				}
			}
			wordVectorHighestTS.put(entry.getKey(), myVector);
			term_scores.add(max);
			term_scores_test.put(max, entry.getKey());
		}
		Collections.sort(term_scores, Collections.reverseOrder());
	}


	/**
	 * @param max_words: The maximum amount of words of a certain lexical class that are allowed in the ontology
	 * @param lexical_class: The lexical class that we will optimize the threshold for, acceptable entries are:
	 * "NN" = noun, "VB" = verb, "JJ" = adjective
	 * @returns The optimal threshold
	 */
	public double create_threshold(int max_words, String lexical_class) throws IOException {
		double threshold_score = 0;
		int n_suggested = 0;
		int n_accepted = 0;
		double opt_treshold_score = Double.NEGATIVE_INFINITY;
		System.out.println(term_scores_test.entrySet());
		Scanner scan = new Scanner(System.in);
		JLanguageTool langTool = new JLanguageTool(new AmericanEnglish());
		for(Map.Entry<Double, String> entry : term_scores_test.entrySet()) {
			String word = entry.getValue(); 
			if(word.contains("#")) {
				word = word.split("#")[0]; 
			}
			List<RuleMatch> matches = langTool.check(word); 

			if (matches.size()==0 && allTerms.get(word) != null) {
				if (allTerms.get(word).contains(lexical_class) && !mention_words.contains(word) && !allAccepted.contains(entry.getValue())) {
					System.out.println("Reject or accept: " +"{"+entry.getValue()+"}" +", This is a " + "{" +allTerms.get(word)+"}"+ ", The TermScore is: " + entry.getKey()+", Press (y) to accept and (n) to reject.");
					String input = scan.nextLine();
					input = input.trim();
					n_suggested += 1;
					if (input.equals("y")) {
						n_accepted += 1;
						System.out.println("accepted!");
					}	
					else if (input.equals("n")) {
						System.out.println("Declined!");
					}

					else {
						boolean error = true;
						while (error) {
							System.out.println("Please enter a valid key");
							System.out.println("Reject or accept: " +"(" +entry.getValue()+")" +"?" +" Press (y) to accept and (n) to reject");
							String input_error = scan.nextLine();
							input_error = input_error.trim();
							if (input_error.equals("y")) {
								n_accepted += 1;
								error = false;
							}
							if (input_error.equals("n")) {
								error = false;
							}
						}
					}
					if (n_accepted > 0) {
						threshold_score = 1/((n_suggested/(double)n_accepted)+(1/(double)n_accepted));
						if (threshold_score > opt_treshold_score){
							opt_treshold_score = threshold_score;
							System.out.println("Optimal score: " + opt_treshold_score + " Number suggested: " + n_suggested + " Number accepted: " + n_accepted);
						}
					}

					if (n_suggested == max_words) {
						break;

					}
				}
			}
			else
			{
				System.out.println("mispelled word found" + " " + entry.getValue());
			}

		}
		System.out.println("Optimal threshold is " + opt_treshold_score);
		return opt_treshold_score;
	}


	/**
	 * 
	 * @param threshold_noun
	 * @param threshold_verb
	 * @param threshold_adj
	 * @param max_noun
	 * @param max_verb
	 * @param max_adj
	 * @throws IOException
	 */
	public HashSet<String> create_term_list(HashSet<String> allTermsSoFar, double threshold_noun, double threshold_verb, double threshold_adj, double threshold_adv, int max_noun, int max_verb, int max_adj, int max_adv) throws IOException {	
		Scanner scan = new Scanner(System.in);
		JLanguageTool langTool = new JLanguageTool(new AmericanEnglish()); 
		int accepted_noun = 0;
		int accepted_verb = 0;
		int accepted_adj = 0;
		int accepted_adv = 0; 
		allAccepted = allTermsSoFar; 

		for (Map.Entry<Double,String> entry : term_scores_test.entrySet()) {
			String word = entry.getValue(); 
			if(word.contains("#")) {
				word = word.split("#")[0]; 
			}
			if(!allTerms.containsKey(word)) {
				word = word.substring(0, word.length()-1); 
			}
			if (!mention_words.contains(word) && !acceptedTerms.containsKey(entry.getValue()) && !allAccepted.contains(entry.getValue())) {
				System.out.println(word);
				List<RuleMatch> matches = langTool.check(word);
				if (matches.size() == 0 && allTerms.containsKey(word)) {
					if (allTerms.get(word).contains("NN") && entry.getKey() > threshold_noun && accepted_noun <= max_noun) {
						accepted_noun = ask_input(entry.getValue(), entry.getKey(), wordVectorHighestTS.get(entry.getValue()), scan, accepted_noun, "noun", langTool);
					}
					else if (allTerms.get(word).contains("VB") && entry.getKey() > threshold_verb && accepted_verb <= max_verb) {
						accepted_verb = ask_input(entry.getValue(), entry.getKey(), wordVectorHighestTS.get(entry.getValue()), scan, accepted_verb, "verb", langTool);
					}
					else if (allTerms.get(word).contains("JJ") && entry.getKey() > threshold_adj && accepted_adj <= max_adj) {
						accepted_adj = ask_input(entry.getValue(), entry.getKey(), wordVectorHighestTS.get(entry.getValue()), scan, accepted_adj, "adj", langTool);
					}
					else if (allTerms.get(word).contains("RB") && entry.getKey() > threshold_adj && accepted_adv <= max_adv) {
						accepted_adj = ask_input(entry.getValue(), entry.getKey(), wordVectorHighestTS.get(entry.getValue()), scan, accepted_adj, "adv", langTool);
					}
				}	
				else
				{
					System.out.println(word + ": misspelled word found, skipping word!");
				}
			}
		}
		System.out.println(acceptedTerms);
		return allAccepted;
	}

	/**
	 * Method to get the nearest words for a given string based on word embeddings
	 * @param word	the string for which to get the i nearest words
	 * @param i	the number of words to get
	 * @return
	 */
	public Map<String, double[]> getNearestWords(String word, double[] vector, int num){  
		Map<String, double[]> similarity_list = new HashMap<>(); 
		Map<String, Double> cosMap = new HashMap<>(); 
		Map<String, double[]> vecMap = new HashMap<>();
		Map<Double, String> treeMap = new TreeMap<Double, String>(new DescOrder()); 
		for(Map.Entry<String, Map<double[], double[]>> entry : finalMap.entrySet()) {
			double highest = -1; 
			double[] myVec = new double[vector.length]; 
			for(double[] vec : entry.getValue().keySet()) {
				double cos = getCosineYelp(vector, vec);
				if(cos > highest) {
					highest = cos; 
					myVec = vec; 
				}
			}
			treeMap.put(highest, entry.getKey());
			cosMap.put(entry.getKey(), highest);
			vecMap.put(entry.getKey(), myVec);
		}
		int i = 0; 
		for(Map.Entry<Double, String> entry : treeMap.entrySet()) {
			String syn = entry.getValue(); 
			if(syn.contains("#")) {
				syn = syn.split("#")[0]; 
			}
			if(i >= num) {
				return similarity_list; 
			}
			if(!similarity_list.containsKey(syn)) {
				similarity_list.put(syn, vecMap.get(entry.getValue())); 
				i++;
			}
		}
		return similarity_list;
	}

	/**
	 * 
	 * @param entry
	 * @param scan
	 * @param to_increase
	 * @param type_word
	 * @param word2vec_yelp
	 * @param langTool
	 * @return
	 * @throws IOException
	 */
	private int ask_input(String word, Double threshold, double[] vector, Scanner scan, int to_increase, String type_word, JLanguageTool langTool) throws IOException {

		boolean error0 = true;
		String input;
		String input2 = null;
		String input3 = null;
		String fixedWord = word; 
		if(word.contains("#")) {
			fixedWord = word.split("#")[0]; 
		}
		if(!allTerms.containsKey(fixedWord)) {
			fixedWord = fixedWord.substring(0, fixedWord.length()-1); 
		}
		while(error0) {
			System.out.println("Reject or accept: " +"{"+ word +"}" +", This is a " + "{" +allTerms.get(fixedWord)+"}"+ ", The TermScore is: " + threshold +", Press (y) to accept and (n) to reject.");
			input = scan.nextLine();
			// Accept the term
			if (input.equals("y")) {

				error0 = false;
				acceptedTerms.put(word, type_word);
				allAccepted.add(word);

				System.out.println("accepted!");
				to_increase += 1;

				// Check if is noun or verb
				if (type_word.equals("noun") || type_word.equals("verb")) {
					boolean error = true;
					// Ask user if it is an AspectMention or SentimentMention 

					while (error) {
						System.out.println("Please indicate whether this is a AspectMention (a) or a Sentiment Mention (s)");
						input2 = scan.nextLine();

						// If is aspectMention do this
						if (input2.equals("a")){
							System.out.println("Added to AspectMentionClass");
							aspect_mentions.put(word, type_word);

							error = false;
						}

						// If is SentimentMention do this
						else if (input2.equals("s")) {
							boolean loop = true;
							while(loop) {
								System.out.println("Is this a type 1,2 or 3 Sentiment Mention? Press (1) for type 1, (2) for type 2, (3) for type 3");
								input3 = scan.nextLine();
								if (input3.equals("1")) {
									sentiment_mentions.put(word, 1);

									loop = false;
								}
								else if (input3.equals("2")) {
									sentiment_mentions.put(word, 2);

									loop = false;
								}
								else if (input3.equals("3")) {
									sentiment_mentions.put(word, 3);

									loop = false;
								}
								else {
									System.out.println("Please input a valid key");
								}
							}
							error = false;
						}

						else {
							System.out.println("Please enter a valid key");
						}
					}

					// wordembedding based adding of similar words
					if (type_word.contentEquals("noun")) {
						Map<String, double[]> similarity_list = getNearestWords(word, vector, 10);
						for (Map.Entry<String, double[]> similarity : similarity_list.entrySet()) {	
							Map<double[], double[]> ptMap = new HashMap<>(); 
							ptMap.put(similarity.getValue(), null); 
							finalMap.put(similarity.getKey(), ptMap);
							if (!mention_words.contains(similarity) && !acceptedTerms.containsKey(similarity) && !allAccepted.contains(similarity)) {
								String simWord = similarity.getKey();
								if(similarity.getKey().contains("#")) {
									simWord =simWord.split("#")[0]; 
								}
								List<RuleMatch> matches = langTool.check(simWord);
								if (getCosineYelp(vector, similarity.getValue()) > 0.7 && matches.size() == 0) {
									if(input2.equals("a") && aspect_mentions.containsKey(similarity)==false) {
										System.out.println("Also added {"+similarity.getKey()+"}");
										aspect_mentions.put(similarity.getKey(), null);	
										acceptedTerms.put(similarity.getKey(), type_word);
										allAccepted.add(similarity.getKey());

									}
									if(input2.equals("s") && sentiment_mentions.containsKey(similarity) == false) {
										if (input3.equals("1")) {
											System.out.println("Also added {"+similarity.getKey()+"}");
											sentiment_mentions.put(similarity.getKey(), 1);
											acceptedTerms.put(similarity.getKey(), type_word);
											allAccepted.add(similarity.getKey());
										}
										else if (input3.equals("2")) {
											System.out.println("Also added {"+similarity.getKey()+"}");
											sentiment_mentions.put(similarity.getKey(), 2);
											acceptedTerms.put(similarity.getKey(), type_word);
											allAccepted.add(similarity.getKey());

										}
										else if (input3.equals("3")) {
											System.out.println("Also added {"+similarity.getKey()+"}");
											sentiment_mentions.put(similarity.getKey(), 3);
											acceptedTerms.put(similarity.getKey(), type_word);
											allAccepted.add(similarity.getKey());

										}		
									}

								}
							}
						}
					}
				}
				if (type_word.equals("adj") || type_word.equals("adv")) 	{
					boolean loop = true;
					while(loop) {
						System.out.println("Is this a type 1,2 or 3 Sentiment Mention? Press (1) for type 1, (2) for type 2, (3) for type3");
						input3 = scan.nextLine();
						if (input3.equals("1")) {
							sentiment_mentions.put(word, 1);

							loop = false;
						}
						else if (input3.equals("2")) {
							sentiment_mentions.put(word, 2);

							loop = false;
						}
						else if (input3.equals("3")) {
							sentiment_mentions.put(word, 3);
							loop = false;
						}
						else {
							System.out.println("Please input a valid key");
						}
					}
				}
			}
			// Decline the term
			else if (input.equals("n")) {
				System.out.println("Declined!");
				error0 = false;
			}
			else
			{
				System.out.println("Please enter a valid key");
			}
		}
		return to_increase;
	}

	public void save_to_file_map_in_map(Map<String, Map<double[], double[]>> myMap, String filelocation) {
		System.out.println("saving file..");
		try {
			File fileOne=new File(filelocation);
			FileOutputStream fos=new FileOutputStream(fileOne);
			ObjectOutputStream oos=new ObjectOutputStream(fos);

			oos.writeObject(myMap);
			oos.flush();
			oos.close();
			fos.close();
		} catch(Exception e) {}
	}

	public void save_to_file_map(Map<String,Integer> terms, String filelocation) {
		System.out.println("saving file..");
		try {
			File fileOne=new File(filelocation);
			FileOutputStream fos=new FileOutputStream(fileOne);
			ObjectOutputStream oos=new ObjectOutputStream(fos);

			oos.writeObject(terms);
			oos.flush();
			oos.close();
			fos.close();
		} catch(Exception e) {}
	}

	public void save_to_file_map_string(Map<String,String> terms, String filelocation) {
		System.out.println("saving file..");
		try {
			File fileOne=new File(filelocation);
			FileOutputStream fos=new FileOutputStream(fileOne);
			ObjectOutputStream oos=new ObjectOutputStream(fos);

			oos.writeObject(terms);
			oos.flush();
			oos.close();
			fos.close(); }
		catch(Exception e) {}
	}

	public int count_synsets(String word) {

		WordNetDatabase wordDatabase = WordNetDatabase.getFileInstance();
		Synset[] synsets = wordDatabase.getSynsets(word);
		if(!allTerms.containsKey(word) && word.length()>0) {
			word = word.substring(0, word.length()-1);
		}
		String pos = allTerms.get(word); 
		int count=0;
		if (synsets.length > 0){
			for (int x = 0; x < synsets.length; x++)
			{
				String type = synsets[x].getType().toString();
				if(pos != null){
					if((type.equals("1") && pos.contains("NN")) || (type.equals("2") && pos.contains("VB")) || (type.equals("3") && pos.contains("JJ")) || (type.equals("4") && pos.contains("RB")))
					{
						count++; 	
					}
				}
			}
		}
		return count; 
	}
	/**
	 * Methode die ook woorden met 1 synsets averaget.
	 * @param myMap
	 * @return
	 */
	public Map<String, Map<double[], double[]>> removeIrrelevantTerms(Map<String, Map<double[], double[]>> myMap){
		Iterator<Map.Entry<String, Map<double[], double[]>>> it = myMap.entrySet().iterator();  
		while(it.hasNext()){
			Map.Entry<String, Map<double[], double[]>> entry = it.next();
			if(irrWords.contains(entry.getKey())) {
				it.remove();
			}
		}
		return myMap; 
	}

	public Map<String, Map<double[], double[]>> averageVectors(Map<String, Map<double[], double[]>> myMap){
		File f = new File(Framework.EXTERNALDATA_PATH + "/WordNet-3.0/dict");
		System.setProperty("wordnet.database.dir", f.toString());

		for(Map.Entry<String, Map<double[], double[]>> entry : myMap.entrySet()) 
		{
			String key = entry.getKey(); 
			Map<double[], double[]> entryMap = entry.getValue(); 
			if(count_synsets(key) == 1) 
			{
				Map.Entry<double[], double[]> entry2 = entryMap.entrySet().iterator().next(); 
				int length = entry2.getKey().length; 
				double[] vectorPT = new double[length];
				double[] vectorFT = new double[length];
				for(int i = 0; i < vectorPT.length; i++) 
				{
					for(double[] vec : entryMap.keySet()) {
						vectorPT[i] += vec[i]; 
					}
					for(double[] vec : entryMap.values()) {
						vectorFT[i] += vec[i]; 
					}
				}
				for(int i = 0; i < vectorPT.length; i++) 
				{
					vectorPT[i] = vectorPT[i] / entryMap.keySet().size();
					vectorFT[i] = vectorFT[i] / entryMap.keySet().size();
				} 
				entryMap.clear(); 
				entryMap.put(vectorPT, vectorFT);
			}	
		}
		return myMap; 
	}

	public void save_outputs(TermSelectionAlgo termSelAlgo){
		termSelAlgo.save_to_file_map_string(termSelAlgo.aspect_mentions, Framework.OUTPUT_PATH + "aspect_mentions");
		termSelAlgo.save_to_file_map(termSelAlgo.sentiment_mentions, Framework.OUTPUT_PATH + "sentiment_mentions");
		termSelAlgo.save_to_file_map_string(termSelAlgo.acceptedTerms, Framework.OUTPUT_PATH + "all_accepted_terms");
		termSelAlgo.save_to_file_map_in_map(termSelAlgo.finalMap, Framework.OUTPUT_PATH + "finalMap");
	}

	public static void main(String args[]) throws Exception {
		// can file location also be the one in repository?		
		String wordEmbeddings = "Roberta.txt";
		String wordEmbeddingsFT = "RobertaFT.txt";
		
		TermSelectionAlgo term_select = new TermSelectionAlgo(Framework.LARGEDATA_PATH + wordEmbeddings, Framework.LARGEDATA_PATH + wordEmbeddingsFT, Framework.OUTPUT_PATH+"Output_stanford_hashmap"); 
		//term_select.createMentionClusters();
		//term_select.create_word_term_score();

		//System.out.println("doing thresholds");
		//double threshold_noun = term_select.create_threshold(100, "NN");
		//double threshold_verb = term_select.create_threshold(20, "VB");
		//double threshold_adj = term_select.create_threshold(80, "JJ");
		//double threshold_adv = term_select.create_threshold(30, "RB");
		//System.out.println("NN= "+ threshold_noun + "VB= "+ threshold_verb + "JJ= "+ threshold_adj + "RB= "+ threshold_adv);
		//term_select.create_term_list(term_select.allAccepted, threshold_noun, threshold_verb, threshold_adj, threshold_adv, 10, 10, 10, 10);

		//term_select.create_term_list(term_select.allAccepted, 0.84, 0.71, 0.81, 0.55, 100, 20, 80, 30);
		//term_select.save_outputs(term_select);

	}
	
	public static void saveJson(Map<String, List<double[]>> myMap, String filelocation) throws IOException {
		try (Writer writer = new FileWriter(filelocation)) {
			Gson gson = new GsonBuilder().create();
			gson.toJson(myMap, writer);
		}
	}
	public static void saveJson2(Map<String, Map<double[], String>> myMap, String filelocation) throws IOException {
		try (Writer writer = new FileWriter(filelocation)) {
			Gson gson = new GsonBuilder().create();
			gson.toJson(myMap, writer);
		}
	}
	static class DescOrder implements Comparator<Double>{
		@Override
		public int compare(Double o1, Double o2) {
			return o2.compareTo(o1);
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


	public static double getCosineYelp(double[] vector, double[] vector2) {
		double ans = dotProduct(vector, vector2) / (getMagnitude(vector) * getMagnitude(vector2)); 
		return ans;
	}


	public void readIrrWords(String filelocation) throws IOException{
		String line; 
		BufferedReader reader = new BufferedReader(new FileReader(filelocation)); 
		while((line = reader.readLine()) != null) {
			irrWords.add(line); 
		}
		reader.close(); 
	}
}


