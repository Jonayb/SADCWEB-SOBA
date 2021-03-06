package seminarOntologyBuilder;


import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.List;
import java.util.Scanner;
import java.util.TreeMap;
import java.util.TreeSet;
import javafx.util.Pair;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.jena.rdf.model.RDFList;
import org.apache.jena.rdf.model.RDFNode;
import org.json.JSONException;
import org.json.JSONObject;

import edu.smu.tspell.wordnet.Synset;
import edu.smu.tspell.wordnet.SynsetType;
import edu.smu.tspell.wordnet.WordNetDatabase;
import edu.smu.tspell.wordnet.impl.file.Morphology;
import edu.eur.absa.Framework;
import edu.eur.absa.data.DatasetJSONReader;
import edu.eur.absa.model.Dataset;
import edu.eur.absa.model.Relation;
import edu.eur.absa.model.Span;
import edu.eur.absa.model.Word;
import edu.eur.absa.model.exceptions.IllegalSpanException;
import seminarOntologyBuilder.Synonyms;
import seminarOntologyBuilder.SkeletalOntology;
import seminarOntologyBuilder.readJSON;
import edu.eur.absa.nlp.*;
import edu.eur.absa.Framework;
import edu.smu.tspell.wordnet.Synset;
import edu.smu.tspell.wordnet.SynsetType;
import edu.smu.tspell.wordnet.WordNetDatabase.*;

//our own classes
import sentimentBuilder.SentimentWordProcessor;
import termSelector.TermSelectionAlgo;
import hierarchicalClustering.clusteringAlgorithm;
import hierarchicalClustering.HierarichalClusterAlgorithm;

//for the clustering things
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.ui.ApplicationFrame;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import com.apporiented.algorithm.clustering.AverageLinkageStrategy;
import com.apporiented.algorithm.clustering.Cluster;
import com.apporiented.algorithm.clustering.ClusteringAlgorithm;
import com.apporiented.algorithm.clustering.DefaultClusteringAlgorithm;
import com.apporiented.algorithm.clustering.ElbowPlotter;
import com.apporiented.algorithm.clustering.visualization.DendrogramFrame;
import com.apporiented.algorithm.clustering.visualization.DendrogramPanel;

/**
 * A method that builds an ontology semi-automatically.
 * 
 * @author Karoliina Ranta
 * Adapted by Fenna ten Haaf
 * Adapted by David van Ommen
 * 
 */
public class OntologyBuilder {

	public final String NS = "http://www.semanticweb.org/bsc.seminar/ontologies/2020/5/Seminar2021Base";
	/* The base ontology. */
	private SkeletalOntology base;
	private HashMap<String, HashSet<String>> aspectCategories;
	private String domain;
	private int numRejectOverall;  //words + parent-relations
	private int numAcceptOverall;
	private HashSet<String> remove;
	private HashSet<String> synonymsAccepted;
	public HashSet<String> allAcceptedTerms; 
	public HashSet<String> acceptedSoFar; 
	public HashMap<String,String> allTermsWithPOS;
	public Map<String, List<double[]>> myPTMap = new HashMap<>(); 
	public Map<String, List<double[]>> myFTMap = new HashMap<>(); 
	private Map<String, Map<double[], double[]>> finalMap; 
	public int maxSimWords = 15; 
	private TermSelectionAlgo synonym_select; 
	/**
	 * A constructor for the OntologyBuilder class.
	 * @param baseOnt, the base ontology from which the final ontology is further constructed
	 * @param aspectCat, the aspect categories of the domain
	 * @param dom, the domain name
	 */
	public OntologyBuilder(SkeletalOntology baseOnt, HashMap<String, HashSet<String>> aspectCat, String dom) throws Exception {

		/* Initialise the base ontology, aspect categories, and domain name. */
		base = baseOnt;
		aspectCategories = aspectCat;
		domain = dom;
		numRejectOverall = 0;
		numAcceptOverall = 0;
		
		remove = new HashSet<String>();
		remove.add("http://www.w3.org/2000/01/rdf-schema#Resource");
		remove.add("http://www.w3.org/2002/07/owl#Thing");
		remove.add(base.URI_Mention);
		remove.add(base.URI_Sentiment);
		remove.add(base.NS + "#" + domain.substring(0, 1).toUpperCase() + domain.substring(1).toLowerCase() + "Mention");
		
		HashMap<String, HashSet<String>> aspectTypes = groupAspects();

		synonymsAccepted = new HashSet<String>();

		HashSet<String> doneAspects = new HashSet<String>();
		allAcceptedTerms = new HashSet<String>();
		
		//We want to start by adding synonyms of particular words to the Generic Positive and Negative classes, to make sure
		//they are included in the ontology
		
		
		synonym_select = new TermSelectionAlgo(Framework.OUTPUT_PATH +"", Framework.OUTPUT_PATH + "" , Framework.OUTPUT_PATH+"Output_stanford_hashmap");//initialise synonyms
		finalMap = synonym_select.finalMap;  
		for(Map.Entry<String, Map<double[], double[]>> entry : finalMap.entrySet()) {
			Map.Entry<double[], double[]> entry2 = finalMap.get(entry.getKey()).entrySet().iterator().next(); 
			List<double[]> PTList = new ArrayList<>();
			List<double[]> FTList = new ArrayList<>(); 
			PTList.add(entry2.getKey());
			FTList.add(entry2.getValue()); 
			myPTMap.put(entry.getKey(), PTList); 
			myFTMap.put(entry.getKey(), FTList);
		}
		
		String positivePropertyURI1 = base.addClass("good#adjective#1", "Good", true, "good", new HashSet<String>(), base.URI_GenericPositiveProperty);
		allAcceptedTerms = this.getSynonymsWithEmbeddings(allAcceptedTerms,"good", maxSimWords , synonym_select, positivePropertyURI1);

		String negativePropertyURI1 = base.addClass("bad#adjective#1", "Bad", true, "bad", new HashSet<String>(), base.URI_GenericNegativeProperty);
		allAcceptedTerms = this.getSynonymsWithEmbeddings(allAcceptedTerms,"bad",maxSimWords, synonym_select, negativePropertyURI1);

		String negativePropertyURI2 = base.addClass("mediocre#adjective#1", "Mediocre", true, "mediocre", new HashSet<String>(), base.URI_GenericNegativeProperty); // eerste mediocre was eerst bad 
		allAcceptedTerms = this.getSynonymsWithEmbeddings(allAcceptedTerms,"mediocre",maxSimWords, synonym_select, negativePropertyURI2);

		String negativePropertyURI3 = base.addClass("expensive#adjective#1", "Expensive", true, "expensive", new HashSet<String>(), base.URI_GenericNegativeProperty);
		allAcceptedTerms = this.getSynonymsWithEmbeddings(allAcceptedTerms,"expensive",maxSimWords, synonym_select, negativePropertyURI3);

		String negativeActionURI1 = base.addClass("hate#verb#1", "Hate", true, "hate", new HashSet<String>(), base.URI_GenericNegativeAction);
		allAcceptedTerms = this.getSynonymsWithEmbeddings(allAcceptedTerms,"hate",maxSimWords, synonym_select,negativeActionURI1);

		String positivePropertyURI2 = base.addClass("great#adjective#1", "Great", true, "great", new HashSet<String>(), base.URI_GenericPositiveProperty);
		allAcceptedTerms = this.getSynonymsWithEmbeddings(allAcceptedTerms,"great", maxSimWords , synonym_select, positivePropertyURI2);

		String positivePropertyURI3 = base.addClass("excellent#adjective#1", "Excellent", true, "excellent", new HashSet<String>(), base.URI_GenericPositiveProperty);
		allAcceptedTerms = this.getSynonymsWithEmbeddings(allAcceptedTerms,"excellent", maxSimWords , synonym_select, positivePropertyURI3);

		String positiveActionURI1 = base.addClass("enjoy#verb#1", "Enjoy", true, "enjoy", new HashSet<String>(), base.URI_GenericPositiveAction);
		allAcceptedTerms = this.getSynonymsWithEmbeddings(allAcceptedTerms,"enjoy",maxSimWords, synonym_select,  positiveActionURI1);

		String positiveActionURI2 = base.addClass("liked#verb#1", "Liked", true, "liked", new HashSet<String>(), base.URI_GenericPositiveAction);
		allAcceptedTerms = this.getSynonymsWithEmbeddings(allAcceptedTerms,"liked",maxSimWords, synonym_select,  positiveActionURI2);

		String positiveModifierURI1= base.addClass("perfectly#adverb#1", "Perfectly", true, "perfectly", new HashSet<String>(), base.URI_GenericPositiveModifier);
		allAcceptedTerms = this.getSynonymsWithEmbeddings(allAcceptedTerms,"perfectly",maxSimWords, synonym_select, positiveModifierURI1);

		String positiveModifierURI2= base.addClass("well#adverb#1", "Well", true, "well", new HashSet<String>(), base.URI_GenericPositiveModifier); 
		allAcceptedTerms = this.getSynonymsWithEmbeddings(allAcceptedTerms,"well",maxSimWords, synonym_select, positiveModifierURI2);

		String negativeModifierURI1 = base.addClass("poorly#adjective#1", "Poorly", true, "poorly", new HashSet<String>(), base.URI_GenericNegativeProperty);
		allAcceptedTerms = this.getSynonymsWithEmbeddings(allAcceptedTerms,"poorly",maxSimWords, synonym_select, negativeModifierURI1);

		String negativeModifierURI2= base.addClass("badly#adverb#1", "Badly", true, "badly", new HashSet<String>(), base.URI_GenericNegativeModifier); 
		allAcceptedTerms = this.getSynonymsWithEmbeddings(allAcceptedTerms,"badly",maxSimWords, synonym_select, negativeModifierURI2);
		
		/* Loop over the aspect category entities. */
		
		
		//create a hashmap with synsets as value of the entities (key), and add as synset property during loop
		HashMap<String, String> entitySynsets = new HashMap<String, String>();
		//add for aspects
		entitySynsets.put("ambience", "ambience#noun#1");
		entitySynsets.put("service", "service#noun#1");
		entitySynsets.put("restaurant", "restaurant#noun#1");
		entitySynsets.put("location", "location#noun#1");
		entitySynsets.put("sustenance", "sustenance#noun#1"); //add drinks and food to sustenance

		for (String entity : aspectCat.keySet()) { 									//for each aspect
			HashSet<String> aspectSet = aspectCat.get(entity); 						//retrieve aspect's categories
			/* Each entity should have its own AspectMention class. */
			HashSet<String> aspects = new HashSet<String>();						// create 'aspect' HashSet, to contain all ASPECT#CATEGORY combinations per aspect
			String synset = entitySynsets.get(entity);								// retrieve synset of aspect
			for (String aspect : aspectSet) {										//per category of an aspect

				/* Don't add miscellaneous to the ontology. */
				if (!aspect.equals("miscellaneous")) {
					aspects.add(entity.toUpperCase() + "#" + aspect.toUpperCase());	// all ASPECT#CATEGORY added to 'aspects'
				}
			}
			String newClassURI = base.addClass(synset, entity.substring(0, 1).toUpperCase() + entity.substring(1).toLowerCase() + "Mention", true, entity, aspects, base.URI_EntityMention);

			/* The domain entity doesn't get sentiment classes. */
			if (!entity.equals(domain)) {

				/* Create the SentimentMention classes (positive and negative) related to the entity. */
				String aspectPropertyClassURI = base.addClass(synset, entity.substring(0, 1).toUpperCase() + entity.substring(1).toLowerCase() + "PropertyMention", true, entity.toLowerCase(), new HashSet<String>(), newClassURI, base.URI_PropertyMention);
				String aspectActionClassURI =  base.addClass(synset, entity.substring(0, 1).toUpperCase() + entity.substring(1).toLowerCase() + "ActionMention", true, entity.toLowerCase(), new HashSet<String>(), newClassURI, base.URI_ActionMention);
				String aspectModifierClassURI =  base.addClass(synset, entity.substring(0, 1).toUpperCase() + entity.substring(1).toLowerCase() + "ModifierMention", true, entity.toLowerCase(), new HashSet<String>(), newClassURI, base.URI_ModifierMention);
				String positivePropertyClassURI = base.addClass(synset, entity.substring(0, 1).toUpperCase() + entity.substring(1).toLowerCase() + "PositiveProperty", false, entity.toLowerCase(), new HashSet<String>(), aspectPropertyClassURI, base.URI_Positive);
				String negativePropertyClassURI = base.addClass(synset, entity.substring(0, 1).toUpperCase() + entity.substring(1).toLowerCase() + "NegativeProperty", false, entity.toLowerCase(), new HashSet<String>(), aspectPropertyClassURI,  base.URI_Negative);
				String positiveActionClassURI = base.addClass(synset, entity.substring(0, 1).toUpperCase() + entity.substring(1).toLowerCase() + "PositiveAction", false, entity.toLowerCase(), new HashSet<String>(), aspectActionClassURI, base.URI_Positive);
				String negativeActionClassURI = base.addClass(synset, entity.substring(0, 1).toUpperCase() + entity.substring(1).toLowerCase() + "NegativeAction", false, entity.toLowerCase(), new HashSet<String>(), aspectActionClassURI, base.URI_Negative);
				String positiveEntityClassURI = base.addClass(synset, entity.substring(0, 1).toUpperCase() + entity.substring(1).toLowerCase() + "PositiveEntity", false, entity.toLowerCase(), new HashSet<String>(), newClassURI, base.URI_Positive);
				String negativeEntityClassURI = base.addClass(synset, entity.substring(0, 1).toUpperCase() + entity.substring(1).toLowerCase() + "NegativeEntity", false, entity.toLowerCase(), new HashSet<String>(), newClassURI, base.URI_Negative);
				String positiveModifierClassURI = base.addClass(synset, entity.substring(0, 1).toUpperCase() + entity.substring(1).toLowerCase() + "PositiveModifier", false, entity.toLowerCase(), new HashSet<String>(), aspectPropertyClassURI, base.URI_Positive);
				String negativeModifierClassURI = base.addClass(synset, entity.substring(0, 1).toUpperCase() + entity.substring(1).toLowerCase() + "NegativeModifier", false, entity.toLowerCase(), new HashSet<String>(), aspectPropertyClassURI,  base.URI_Negative);
			} 

			/* Create AspectMention and SentimentMention subclasses for all aspects except for general and miscellaneous. */
			for (String aspectName : aspectTypes.keySet()) { // for each category
				if (!aspectName.equals("general") && !aspectName.equals("miscellaneous") && !doneAspects.contains(aspectName)) {
					doneAspects.add(aspectName);

					/* Create the AspectMention class. */
					HashSet<String> aspectsAsp = new HashSet<String>();
					for (String entityName : aspectTypes.get(aspectName)) { // retrieve all aspects per category
						aspectsAsp.add(entityName.toUpperCase() + "#" + aspectName.toUpperCase()); //ASPECT#CATEGORY added to aspectAsp, to be added to class as aspect-property
					}
					//add CategoryMention class
					String newClassURIAspect = base.addClass(aspectName.substring(0, 1).toUpperCase() + aspectName.substring(1).toLowerCase() + "Mention", true, aspectName, aspectsAsp, base.URI_EntityMention);

					/* Create the SentimentMention classes. */
					String aspectPropertyClassURI = base.addClass(aspectName.substring(0, 1).toUpperCase() + aspectName.substring(1).toLowerCase() + "PropertyMention", true, entity.toLowerCase(), new HashSet<String>(), newClassURIAspect, base.URI_PropertyMention);
					String aspectActionClassURI =  base.addClass(aspectName.substring(0, 1).toUpperCase() + aspectName.substring(1).toLowerCase() + "ActionMention", true, entity.toLowerCase(), new HashSet<String>(), newClassURIAspect, base.URI_ActionMention);
					String aspectModifierClassURI =  base.addClass(aspectName.substring(0, 1).toUpperCase() + aspectName.substring(1).toLowerCase() + "ModifierMention", true, entity.toLowerCase(), new HashSet<String>(), newClassURIAspect, base.URI_ModifierMention);
					String positivePropertyURI = base.addClass(aspectName.substring(0, 1).toUpperCase() + aspectName.substring(1).toLowerCase() + "PositiveProperty", false, aspectName.toLowerCase(), new HashSet<String>(), aspectPropertyClassURI, base.URI_Positive);
					String negativePropertyURI = base.addClass(aspectName.substring(0, 1).toUpperCase() + aspectName.substring(1).toLowerCase() + "NegativeProperty", false, aspectName.toLowerCase(), new HashSet<String>(), aspectPropertyClassURI, base.URI_Negative);
					String positiveActionURI = base.addClass(aspectName.substring(0, 1).toUpperCase() + aspectName.substring(1).toLowerCase() + "PositiveAction", false, aspectName.toLowerCase(), new HashSet<String>(), aspectActionClassURI, base.URI_Positive);
					String negativeActionURI = base.addClass(aspectName.substring(0, 1).toUpperCase() + aspectName.substring(1).toLowerCase() + "NegativeAction", false, aspectName.toLowerCase(), new HashSet<String>(), aspectActionClassURI, base.URI_Negative);
					String positiveEntityURI = base.addClass(aspectName.substring(0, 1).toUpperCase() + aspectName.substring(1).toLowerCase() + "PositiveEntity", false, aspectName.toLowerCase(), new HashSet<String>(), newClassURIAspect, base.URI_Positive);
					String negativeEntityURI = base.addClass(aspectName.substring(0, 1).toUpperCase() + aspectName.substring(1).toLowerCase() + "NegativeEntity", false, aspectName.toLowerCase(), new HashSet<String>(), newClassURIAspect, base.URI_Negative);					
					String positiveModifierURI = base.addClass(aspectName.substring(0, 1).toUpperCase() + aspectName.substring(1).toLowerCase() + "PositiveModifier", false, aspectName.toLowerCase(), new HashSet<String>(), newClassURIAspect, base.URI_Positive);
					String negativeModifierURI = base.addClass(aspectName.substring(0, 1).toUpperCase() + aspectName.substring(1).toLowerCase() + "NegativeModifier", false, aspectName.toLowerCase(), new HashSet<String>(), newClassURIAspect,  base.URI_Negative);

					if (aspectName.contains("&")) {
						HashSet<String> lexs = new HashSet<String>();
						String[] parts = aspectName.split("&");
						lexs.add(parts[0]);
						lexs.add(parts[1]);
						base.addLexicalizations(newClassURIAspect, lexs);
						base.addLexicalizations(aspectPropertyClassURI, lexs);
						base.addLexicalizations(aspectActionClassURI, lexs);
					}
					if (aspectName.contains("_")) { 
						HashSet<String> lexs = new HashSet<String>();
						String[] parts = aspectName.split("_");
						lexs.add(parts[0]);
						lexs.add(parts[1]);
						base.addLexicalizations(newClassURIAspect, lexs);
						base.addLexicalizations(aspectPropertyClassURI, lexs);
						base.addLexicalizations(aspectActionClassURI, lexs);
					}
				}
			}			
		}


		//add Food and DrinksMention to Sustenance Class

		String FoodMentionClassURI = base.addClass("food#noun#1", "FoodMention",true, "food", aspectCat.get("sustenance"), base.NS + "#SustenanceMention");
		String FoodMentionActionClassURI = base.addClass("food#noun#1", "FoodActionMention",true, "food", aspectCat.get("sustenance"), base.NS + "#SustenanceActionMention");
		String FoodMentionPropertyClassURI = base.addClass("food#noun#1",  "FoodPropertyMention", true, "food", aspectCat.get("sustenance"), base.NS + "#SustenancePropertyMention");
		String FoodMentionModifierClassURI = base.addClass("food#noun#1",  "FoodModifierMention", true, "food", aspectCat.get("sustenance"), base.NS + "#SustenanceModifierMention");
		
		String foodPositivePropertyURI = base.addClass("FoodPositiveProperty", false, "food", new HashSet<String>(), FoodMentionPropertyClassURI, base.URI_Positive);
		String foodNegativePropertyURI = base.addClass("FoodNegativeProperty", false, "food", new HashSet<String>(), FoodMentionPropertyClassURI, base.URI_Negative);
		String foodPositiveActionURI = base.addClass("FoodPositiveAction", false, "food", new HashSet<String>(), FoodMentionActionClassURI, base.URI_Positive);
		String foodNegativeActionURI = base.addClass("FoodNegativeAction", false, "food", new HashSet<String>(), FoodMentionActionClassURI, base.URI_Negative);
		String foodPositiveEntityURI = base.addClass("FoodPositiveEntity", false, "food", new HashSet<String>(), FoodMentionClassURI, base.URI_Positive);
		String foodNegativeEntityURI = base.addClass("FoodNegativeEntity", false, "food", new HashSet<String>(), FoodMentionClassURI, base.URI_Negative);					
		String foodPositiveModifierURI = base.addClass("FoodPositiveModifier", false, "food", new HashSet<String>(), FoodMentionPropertyClassURI, base.URI_Positive);
		String foodNegativeModifierURI = base.addClass("FoodNegativeModifier", false, "food", new HashSet<String>(), FoodMentionPropertyClassURI,  base.URI_Negative);

		String DrinksMentionClassURI = base.addClass("drinks#noun#1", "DrinksMention", true, "drinks", aspectCat.get("sustenance"), base.NS + "#SustenanceMention"); // als het niet werkt deze hashset leeg laten
		String DrinksMentionActionClassURI = base.addClass("drinks#noun#1", "DrinksActionMention", true, "drinks", aspectCat.get("sustenance"), base.NS + "#SustenanceActionMention");
		String DrinksMentionPropertyClassURI = base.addClass("drinks#noun#1", "DrinksPropertyMention", true, "drinks", aspectCat.get("sustenance"), base.NS + "#SustenancePropertyMention");
		String DrinksMentionModifierClassURI = base.addClass("drinks#noun#1", "DrinksModifierMention", true, "drinks", aspectCat.get("sustenance"), base.NS + "#SustenanceModifierMention");
		
		String drinksPositivePropertyURI = base.addClass("DrinksPositiveProperty", false, "drinks", new HashSet<String>(), DrinksMentionPropertyClassURI, base.URI_Positive);
		String drinksNegativePropertyURI = base.addClass("DrinksNegativeProperty", false, "drinks", new HashSet<String>(), DrinksMentionPropertyClassURI, base.URI_Negative);
		String drinksPositiveActionURI = base.addClass("DrinksPositiveAction", false, "drinks", new HashSet<String>(), DrinksMentionActionClassURI, base.URI_Positive);
		String drinksNegativeActionURI = base.addClass("DrinksNegativeAction", false, "drinks", new HashSet<String>(), DrinksMentionActionClassURI, base.URI_Negative);
		String drinksPositiveEntityURI = base.addClass("DrinksPositiveEntity", false, "drinks", new HashSet<String>(), DrinksMentionClassURI, base.URI_Positive);
		String drinksNegativeEntityURI = base.addClass("DrinksNegativeEntity", false, "drinks", new HashSet<String>(), DrinksMentionClassURI, base.URI_Negative);					
		String drinksPositiveModifierURI = base.addClass("DrinksPositiveModifier", false, "drinks", new HashSet<String>(), DrinksMentionPropertyClassURI, base.URI_Positive);
		String drinksNegativeModifierURI = base.addClass("DrinksNegativeModifier", false, "drinks", new HashSet<String>(), DrinksMentionPropertyClassURI,  base.URI_Negative);
		//add a few extra EntityMention classes 

		//ExperienceMention
		HashSet<String> experienceAspects = new HashSet<String>();
		experienceAspects.add("RESTAURANT#MISCELLANEOUS");
		
		String ExperienceMentionClassURI = base.addClass("experience#noun#3", "Experience" + "Mention", true, "experience", experienceAspects, base.URI_EntityMention);
		String ExperienceMentionActionClassURI = base.addClass("experience#noun#3", "Experience" + "ActionMention", true, "experience", experienceAspects, base.URI_ActionMention);
		String ExperienceMentionPropertyClassURI = base.addClass("experience#noun#3", "Experience" + "PropertyMention", true, "experience", experienceAspects, base.URI_PropertyMention);
		String ExperienceMentionModifierClassURI = base.addClass("experience#noun#3", "Experience" + "ModifierMention", true, "experience", experienceAspects, base.URI_ModifierMention);
		//Lastly, add some specific adjectives to put in

		//Food adjectives
		String positiveFoodURI1 = base.addClass("tasty#adjective#1", "Tasty", true, "tasty", new HashSet<String>(), base.NS + "#FoodPositiveProperty");
		allAcceptedTerms = this.getSynonymsWithEmbeddings(allAcceptedTerms, "tasty",maxSimWords, synonym_select,  positiveFoodURI1);
	
	}

	/**
	 * A method to perform the termselection
	 */
	public void getTerms() throws Exception 
	{
		//TermSelectionAlgo term_select = new TermSelectionAlgo(Framework.OUTPUT_PATH+ "clstrd500RestaurantFT", Framework.OUTPUT_PATH + "clstrd500ContrastingPreT", Framework.OUTPUT_PATH+"Output_stanford_hashmap");
		synonym_select.create_word_term_score();
		
		System.out.println("doing thresholds");
		boolean known = true; // When the optimal thresholds are known, set true, else false
		
		double threshold_noun;
		double threshold_verb;
		double threshold_adj;
		double threshold_adv;
		if (!known) {
			threshold_noun = synonym_select.create_threshold(100, "NN");
			threshold_verb = synonym_select.create_threshold(20, "VB");
			threshold_adj = synonym_select.create_threshold(80, "JJ");
			threshold_adv = synonym_select.create_threshold(20, "RB");
		} else {
			threshold_noun = 0.84;
			threshold_verb = 0.71;
			threshold_adj = 0.81;
			threshold_adv = 0.55;
		}
		
		System.out.println(threshold_noun + " " + threshold_verb + " " + threshold_adj + " " + threshold_adv);
		
		allAcceptedTerms = synonym_select.create_term_list(allAcceptedTerms, threshold_noun, threshold_verb, threshold_adj, threshold_adv, 100, 20, 80, 30); 
		synonym_select.save_outputs(synonym_select);
	}


	/**
	 * Method to create a hierarchy within the sentiment words, and add them to the ontology
	 * @throws IOException
	 * @throws ClassNotFoundException
	 */
	public void addSentimentWords() throws IOException, ClassNotFoundException
	{

		//First, we get the hashMap that will give us whether we have a noun or a verb 
		File toRead_terms=new File(Framework.OUTPUT_PATH+ "Output_stanford_hashmap200k");
		FileInputStream fis_terms=new FileInputStream(toRead_terms);
		ObjectInputStream ois_terms =new ObjectInputStream(fis_terms);
		allTermsWithPOS =(HashMap<String,String>)ois_terms.readObject();
		ois_terms.close();
		fis_terms.close();	

		HashSet<String> set;

		//Get the clustered sentiment words
		Map<String, Map<String,String>> clustered_sentiment = new HashMap<String,Map<String,String>>();
		SentimentWordProcessor sent_calc = new SentimentWordProcessor(Framework.OUTPUT_PATH+ "finalMap", Framework.OUTPUT_PATH + "sentiment_mentions");
		clustered_sentiment = sent_calc.create_sentiment_links();

		for (Map.Entry<String,Map<String,String>> entry : clustered_sentiment.entrySet()) 
		{
			String mentionClass = entry.getKey(); 
			Map<String,String> sentPolarities = entry.getValue();

			for (Map.Entry<String,String> entry2 : sentPolarities.entrySet()) 
			{
				String sentWord = entry2.getKey(); // the word to add to our ontology
				if(sentWord.contains("#")) {
					sentWord = sentWord.split("#")[0]; 
				}
				String pol = entry2.getValue(); // the polarity class
				String pos = ""; 
				
				if(!allTermsWithPOS.keySet().contains(sentWord)) {
					sentWord = sentWord.substring(0, sentWord.length()-1); 
				}
				//Define the part-of-speech
				if (allTermsWithPOS.get(sentWord) == null) {
					continue;
				}
				if (allTermsWithPOS.get(sentWord).contains("NN"))
				{
					pos = "noun";

					set = base.getSubclasses(base.URI_EntityMention); 
					set.remove(base.URI_EntityMention);
					HashSet<String> temp = base.getSubclasses(base.URI_ActionMention);
					temp.addAll(base.getSubclasses(base.URI_PropertyMention));
					temp.addAll(base.getSubclasses(base.URI_Sentiment));
					set.removeAll(temp);
				}
				else if (allTermsWithPOS.get(sentWord).contains("VB"))
				{
					pos = "verb"; 

					set = base.getSubclasses(base.URI_ActionMention); 
					set.remove(base.URI_ActionMention);
					HashSet<String> temp = base.getSubclasses(base.URI_Sentiment);
					set.removeAll(temp);
				}	
				else if (allTermsWithPOS.get(sentWord).contains("JJ"))
				{
					pos = "adjective"; 	

					set = base.getSubclasses(base.URI_PropertyMention); 
					set.remove(base.URI_PropertyMention);
					HashSet<String> temp = base.getSubclasses(base.URI_Sentiment);
					set.removeAll(temp);
				}
				else {
					pos = "adverb"; 
					set = base.getSubclasses(base.URI_ModifierMention); 
					set.remove(base.URI_ModifierMention);
					HashSet<String> temp = base.getSubclasses(base.URI_Sentiment);
					set.removeAll(temp);	
				}
				String parentClassURI = "";

				if(pol.equals("positive")) //positive polarity
				{ 
					if (mentionClass.equals("generic")) { // add to type 1 positive entity, action or property mention
						if (pos.equals("noun"))
						{
							parentClassURI = base.URI_GenericPositiveEntity;
						}
						else if (pos.equals("verb"))
						{			
							parentClassURI = base.URI_GenericPositiveAction;
						}	
						else if (pos.equals("adjective"))
						{
							parentClassURI = base.URI_GenericPositiveProperty;
						}
						else if (pos.equals("adverb"))
						{
							parentClassURI = base.URI_GenericPositiveModifier;
						}
					}
					else // add to other type of negative entity, action or property mention
					{
						if (pos.equals("noun"))
						{
							parentClassURI = NS + "#" + mentionClass.substring(0, 1).toUpperCase() + mentionClass.substring(1).toLowerCase() + "PositiveMention";
						}
						else if (pos.equals("verb"))
						{			
							parentClassURI = NS + "#" + mentionClass.substring(0, 1).toUpperCase() + mentionClass.substring(1).toLowerCase()+ "PositiveAction";
						}	
						else if (pos.equals("adjective"))
						{
							parentClassURI = NS + "#" + mentionClass.substring(0, 1).toUpperCase() + mentionClass.substring(1).toLowerCase()+ "PositiveProperty";
						}
						else if (pos.equals("adverb"))
						{
							parentClassURI = NS + "#" + mentionClass.substring(0, 1).toUpperCase() + mentionClass.substring(1).toLowerCase()+ "PositiveModifier";
						}
					}
				}
				else // negative polarity
				{  
					if (mentionClass.equals("generic")) { // add to type 1 positive entity, action or property mention
						if (pos.equals("noun"))
						{
							parentClassURI = base.URI_GenericNegativeEntity;
						}
						else if (pos.equals("verb"))
						{			
							parentClassURI = base.URI_GenericNegativeAction;
						}	
						else if (pos.equals("adjective"))
						{
							parentClassURI = base.URI_GenericNegativeProperty;
						}
						else if (pos.equals("adverb"))
						{
							parentClassURI = base.URI_GenericNegativeModifier;
						}
					}
					else // add to other type of negative entity, action or property mention
					{
						if (pos.equals("noun"))
						{
							parentClassURI = NS + "#"+ mentionClass.substring(0, 1).toUpperCase() + mentionClass.substring(1).toLowerCase()+ "NegativeMention";
						}
						else if (pos.equals("verb"))
						{			
							parentClassURI = NS +  "#" + mentionClass.substring(0, 1).toUpperCase() + mentionClass.substring(1).toLowerCase()+ "NegativeAction";
						}	
						else if (pos.equals("adjective"))
						{
							parentClassURI = NS + "#" + mentionClass.substring(0, 1).toUpperCase() + mentionClass.substring(1).toLowerCase()+ "NegativeProperty";
						}
						else if (pos.equals("adverb"))
						{
							parentClassURI = NS + "#" + mentionClass.substring(0, 1).toUpperCase() + mentionClass.substring(1).toLowerCase()+ "NegativeModifier";
						}
					}

				}

				//Now we add the new parent URI to the ontology
				base.addClass(pos, sentWord.substring(0, 1).toUpperCase() + sentWord.substring(1).toLowerCase(), true, sentWord, new HashSet<String>(), parentClassURI);
			} 
		}
	}

	/**
	 * Method to create the hierarchies within the aspect mentions and add them to the ontology
	 * @throws Exception
	 */
	public void getHierarchicalClusters() throws Exception {

		//Now start the method
		Scanner scanner = new Scanner(System.in);

		String[] mentionclasses = {"restaurant","ambience","service","Location","food","drinks","price","quality","style","options","experience"};
		int numberofclusters = mentionclasses.length;
		int iterations = 100;
		String name = "aspect_mentions";
		String approach = "similarities";

		clusteringAlgorithm HC = new clusteringAlgorithm(name, numberofclusters, iterations, mentionclasses, approach);
		List<Pair<String,Map<String,List<String>>>> clusterRepPairList = HC.getHierarchicalClusters();


		for (Pair<String,Map<String,List<String>>> clusterRepPair : clusterRepPairList)
		{

			String mentClass = clusterRepPair.getKey();
			Map<String,List<String>> clusterRepresentation = clusterRepPair.getValue();

			for (Map.Entry<String, List<String>> entry2 : clusterRepresentation.entrySet()) {
				// parent-child relation
				String parent = entry2.getKey();
				List<String> children = entry2.getValue();

				if (!children.isEmpty()) {

					// Now we add the parent-child relation to the skeletal ontology
					for (String child : children) {

						String checkChild ="";
						String checkParent ="";

						// Some of the clusters are labeled as just clstr#--, this needs to be fixed.
						// For now however, we just check if the parent or child cluster do not have 
						// clstr# in their name
						if(child.length()>5)
						{
							checkChild = child.substring(0,5);
						}
						else
						{
							checkChild = "it's fine!"; // if it is shorter, then it can't be clstr#

						}

						if(parent.length()>5)
						{
							checkParent = parent.substring(0,5);
						}
						else
						{
							checkParent = "it's fine!"; // if it is shorter, then it can't be clstr# 

						}

						String parentClassURI="";

						if(!(checkChild.equals("clstr")) && !(checkParent.equals("clstr"))) // no clstr# in the name
						{
							if(child.contains("#")) {
								child = child.split("#")[0]; 
							}
							if(parent.contains("#")) {
								parent = parent.split("#")[0];
							}
							System.out.println("Parent: "+parent+" and child: "+child+ " in the MentionClass: "+mentClass);

							String pos = "noun"; 
							// We treat all the cluster names as nouns, as this is the most likely (only alternative is verb,
							// adjectives are sentiment mentions) and it does not work otherwise

							if (parent.equals(child)) { // in this case, we are talking about a direct subclass of mentionClass
								parentClassURI = NS + "#"+ mentClass.substring(0, 1).toUpperCase() + mentClass.substring(1).toLowerCase()+ "Mention";
							}
							else {
								parentClassURI = NS + "#"+ parent.substring(0, 1).toUpperCase() + parent.substring(1).toLowerCase()+ "Mention";
							}

							String newConcept = base.addClass(pos, child.substring(0, 1).toUpperCase() + child.substring(1).toLowerCase(), true, child, new HashSet<String>(), parentClassURI);
						}

						// if the parent name is just "clstr#-- we do not know how to handle this yet,
						// so we just add the child as a direct subclass
						if((checkParent.equals("clstr"))) 
						{ 
							if(child.contains("#")) {
								child = child.split("#")[0]; 
							}
							String pos = "noun"; // we treat it as a noun

							parentClassURI = NS + "#"+ mentClass.substring(0, 1).toUpperCase() + mentClass.substring(1).toLowerCase()+ "Mention";
							String newConcept = base.addClass(pos, child.substring(0, 1).toUpperCase() + child.substring(1).toLowerCase(), true, child, new HashSet<String>(), parentClassURI);
						}
					}
				}
			}
		}
	}


	/**
	 * A method to get a number of similar words using word embeddings
	 * @param word
	 * @throws Exception
	 */
	public HashSet<String> getSynonymsWithEmbeddings(HashSet<String>acceptedSoFar, String word, int synonymNum, TermSelectionAlgo synsel, String... classURI) throws Exception{
		HashSet<String> accepted = new HashSet<String>();
		HashSet<String> rejected = new HashSet<String>();
		HashSet<String> acceptSoFar = acceptedSoFar; 
		acceptSoFar.add(word);
		List<double[]> temp = new ArrayList<>();
		if(myPTMap.containsKey(word)) {
			Integer numAccepted = 0; 
			int SYNONYM_NUM = synonymNum; 
			TermSelectionAlgo synonym_select = synsel;
			int length = 0; 
			// Add the word that we want synonyms of to the accepted terms list as well
			for(Map.Entry<String, List<double[]>> entry : myPTMap.entrySet()) {
				length = entry.getValue().get(0).length; 
			}
			double[] toAverage = new double[length]; 
			temp = myPTMap.get(word); 
			for(int i = 0; i < length; i++) {
				for(double[] vec : temp) {
					toAverage[i] += vec[i]; 
				}
			}
			for(int i = 0; i < toAverage.length; i++) {
				toAverage[i] = toAverage[i] / temp.size();
			}
			Map<String, double[]> similar_words_list = synonym_select.getNearestWords(word, toAverage, SYNONYM_NUM); 
			System.out.println("Enter 'a' to accept and 'r' to reject the synonym: " );
			Scanner input = new Scanner(System.in);
			int i = 0;
			for (String nearTerm : similar_words_list.keySet()) {
				i++;
				String newTerm = nearTerm; 
				if (nearTerm.equals(word) || accepted.contains(newTerm) || rejected.contains(newTerm) || acceptSoFar.contains(newTerm))  {
					continue; //in this case, we have already suggested the term. we won't suggest it again.
				}
				while(true) {
					System.out.println("synonym: " + word + " --> " + nearTerm);
					String userInput = input.next();
					if (userInput.equals("a")) {
						numAccepted++;
						numAcceptOverall++;
						accepted.add(newTerm);
						synonymsAccepted.add(newTerm);
						acceptSoFar.add(newTerm);
						break;
					} 
					else if (userInput.equals("r")) {
						rejected.add(newTerm);
						numRejectOverall++;
						break; 
					}
					else {
						System.out.print("Please type either a or r."+'\n');
					}
				}
			}
			for (String URI : classURI) {
				base.addLexicalizations(URI, accepted);
			}
		}
		return acceptSoFar;
	}


	/**
	 * A method that suggests the synonyms of a word and adds it as a lexicalization to the concepts, using wordNet.
	 * @param classURI, the concepts to which to add the lexicalizations
	 * @param word, the word of which to find synonyms
	 */

	
	public void suggestSynonyms(String word, String... classURI) {
		HashSet<String> accepted = new HashSet<String>();
		HashSet<String> rejected = new HashSet<String>();
		Integer numAccepted = 0; 
		Synonyms syn = new Synonyms(word);


		// Add the word that we want synonyms of to the accepted terms list as well
		allAcceptedTerms.add(word);

		System.out.println("Enter 'a' to accept and 'r' to reject the synonym: ");
		Scanner input = new Scanner(System.in);
		int i = 0;
		for (String synonym : syn.synonyms()) {
			i++;
			if (i > 20 || numAccepted>5) { //we stop if we have suggested more than this number of synonyms or if we already have 5 synonyms
				break; 
			}

			if (synonym.equals(word) || accepted.contains(synonym) || rejected.contains(synonym) || allAcceptedTerms.contains(synonym))  {
				continue; //in this case, we have already suggested the term. we won't suggest it again.
			}

			while(true) {
				System.out.println("synonym: " + word + " --> " + synonym);
				String userInput = input.next();
				if (userInput.equals("a")) {
					numAccepted++;
					numAcceptOverall++;
					accepted.add(synonym);
					synonymsAccepted.add(synonym);
					allAcceptedTerms.add(synonym);
					break;

				} 
				else if (userInput.equals("r")) {
					rejected.add(synonym);
					numRejectOverall++;
					break; 
				}
				else {
					System.out.print("Please type either a or r."+'\n');
				}
			}
		}
		for (String URI : classURI) {
			base.addLexicalizations(URI, accepted);
		}
	}
	 


	/**
	 * Creates an object that stores all the aspect types and for each aspect which entities have this aspect.
	 * @return The HashMap containing the aspects and corresponding entities.
	 */
	public HashMap<String, HashSet<String>> groupAspects() {
		HashMap<String, HashSet<String>> aspectTypes = new HashMap<String, HashSet<String>>();

		/* Loop over the entities. */
		for (String entity : aspectCategories.keySet()) {

			/* Loop over the aspects of the entity. */
			for (String aspect : aspectCategories.get(entity)) {
				HashSet<String> entities;

				/* Check if the set already contains the aspect. */
				if (aspectTypes.containsKey(aspect)) {
					entities = aspectTypes.get(aspect);
				} else {
					entities = new HashSet<String>();
				}
				entities.add(entity);
				aspectTypes.put(aspect, entities);
			}
		}
		return aspectTypes;
	}

	/**
	 * A method that returns the number of accepted and rejected terms.
	 * @return an array with first the number of accepted and second number of rejected terms
	 */
	public int[] getStats() {
		int[] stats = new int[3];
		stats[0] = numAcceptOverall; //change to numAcceptTerms if you only need words and no parent-relations
		stats[1] = numRejectOverall;
		return stats;
	}

	/**
	 * converts a verb to its most simple form.
	 * @param verb to be converted
	 * @return simple form of the verb
	 */
	public String verbConvertion(String verb) {

		System.setProperty("wordnet.database.dir", Framework.EXTERNALDATA_PATH + "WordNet-3.0/dict/");
		WordNetDatabase database = WordNetDatabase.getFileInstance();

		Morphology id = Morphology.getInstance();

		String[] arr = id.getBaseFormCandidates(verb, SynsetType.VERB);
		if (arr.length>0) {
			return arr[0];
		}
		return verb;

	}

	/**
	 * Save the built ontology.
	 * @param file, the name of the file to which to save the ontology
	 */
	public void save(String file) {
		base.save(file);
	}
}