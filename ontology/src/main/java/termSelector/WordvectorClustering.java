package termSelector;

import java.awt.Frame;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.swing.JLabel;

import org.apache.commons.lang3.ArrayUtils;
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

import edu.eur.absa.Framework;
public class WordvectorClustering {

	private List<double[]> myList; 
	private String[] terms; 
	private Map<String, double[]> term_wordvector = new HashMap<>(); 
	private Map<double[], String> wordvector_term = new HashMap<>(); 
	private List<double[]> vector_list = new ArrayList<double[]>();
	private Map<Integer,Double> elbow_data = new HashMap<Integer, Double>();
	public Map<String,List<String>> cluster_representation = new HashMap<>();


	public WordvectorClustering(List<double[]> myList, String[] terms) throws ClassNotFoundException, IOException {
		this.myList = myList; 
		this.terms = terms; 
		create_term_wordvec(); 
		create_wordvec_term(); 
	}
	
	public Map<String, double[]> get_term_wordvector() {
		return term_wordvector; 
	}
	
	public void create_term_wordvec(){
		for(int i = 0; i < terms.length; i++) { 
		
			if(myList.get(i) != null) {
				term_wordvector.put(terms[i], myList.get(i));
			}
			else {
				terms = ArrayUtils.removeElement(terms, terms[i]);
			}
		}
	}
	
	public void create_wordvec_term() {
		for(int i = 0; i < terms.length; i++) {
			if(myList.get(i) != null) {
				wordvector_term.put(myList.get(i), terms[i]); 
			}
			else {
				terms = ArrayUtils.removeElement(terms, terms[i]);
			}
		}
	}
	/**
	{
		File toRead_yelp=new File(filelocation_yelp);
		FileInputStream fis_yelp=new FileInputStream(toRead_yelp);
		ObjectInputStream ois_yelp =new ObjectInputStream(fis_yelp);
		myMap =(HashMap<String,List<double[]>>)ois_yelp.readObject();
		ois_yelp.close();
		fis_yelp.close();	
	}
	**/
	

	/**
	public static void clusterWords(Map<String, List<double[]>> myMap) {
		for(Map.Entry<String, List<double[]>> entry : myMap.entrySet()) {
			double[][] distances = getDistanceMatrix(entry.getValue()); 
			ClusteringAlgorithm alg = new DefaultClusteringAlgorithm(); 
			String[] terms = new String[entry.getValue().size()]; 
			for(int i = 0; i < entry.getValue().size(); i++) {
				terms[i] = String.valueOf(i);
			}
			Cluster cluster = alg.performClustering(distances, terms, new AverageLinkageStrategy());
			int recursion = recursion_depth(cluster);
			//System.out.println(cluster);
		}
	}
	**/
	public double getDistance(double[] vec1, double[] vec2) {
		double distance = 0;
		for (int i = 0; i< vec1.length; i++){
			distance += Math.pow(vec1[i]-vec2[i],2);
		}
		return Math.sqrt(distance);
	}

	public double[][] getDistanceMatrix(List<double[]> listWordvectors) {
		int totalterms = listWordvectors.size();
		double[][] distancematrix = new double[totalterms][totalterms];
		for (int i = 0; i < totalterms; i++) {
			for (int j = 0; j < totalterms; j++) {
				double[] rowterm = listWordvectors.get(i);
				double[] columnterm = listWordvectors.get(j);
				double distance = getDistance(rowterm, columnterm);
				distancematrix[i][j] = distance;
			}
		}
		return distancematrix;
	}

	public int recursion_depth(Cluster cluster) {
		if (!cluster.isLeaf()){
			List<Cluster> child_clusters = cluster.getChildren();
			if (child_clusters.size() == 1) {
				return recursion_depth(cluster) + 1;
			}
			if (child_clusters.size() == 2) {
				return Math.max(recursion_depth(child_clusters.get(0)), recursion_depth(child_clusters.get(1))) + 1;
			}
		}

		return 0;
	}
	
	public List<double[]> get_vector_list(Cluster cluster, List<double[]> vector_list){
	
		if (cluster.isLeaf() && term_wordvector.get(cluster.getName()) != null){ 
			
			vector_list.add(term_wordvector.get(cluster.getName()));		
		}
		else {	
			List<Cluster> child_clusters = cluster.getChildren();
			if (child_clusters.size() == 1) {
				get_vector_list(child_clusters.get(0), vector_list);
			}
			if (child_clusters.size() == 2) {
				get_vector_list(child_clusters.get(0), vector_list);
				get_vector_list(child_clusters.get(1), vector_list);
			}
		}
		return vector_list;

	}


	public double calculate_WSS(Cluster cluster) { 
		List<double[]> vector_list = new ArrayList<double[]>();
		List<double[]> vectors_list = get_vector_list(cluster, vector_list);
		double [] average_vector;

		//if (!vectors_list.isEmpty())
		//{
			average_vector= new double[vectors_list.get(0).length];
		

		for( int i = 0; i<vectors_list.size(); i++) { 
			for (int j = 0; j < vectors_list.get(0).length; j ++) {
				average_vector[j] += vectors_list.get(i)[j];
				if (i == vectors_list.size() - 1) {
					average_vector[j] = average_vector[j] / (double)vectors_list.size();
				}
			}
		}
		double WSS = 0;
		for (int x = 0; x < vectors_list.size(); x++) {
			for (int z = 0; z < vectors_list.get(x).length; z ++) {
				WSS += Math.pow(vectors_list.get(x)[z]-average_vector[z], 2);
			}
		}
		return WSS;
		//}
		
		/*
		else
		{
			return 0; 
		}
		*/
		
	}


	public double get_WSS(int total_depth, int max_depth, Cluster cluster) {
		if (total_depth < max_depth) {
			List<Cluster> child_clusters = cluster.getChildren();
			if (child_clusters.size() == 1) {
				total_depth += 1;
				return get_WSS(total_depth, max_depth, child_clusters.get(0));
			}
			if (child_clusters.size() == 2) {
				total_depth += 1;
				return get_WSS(total_depth, max_depth, child_clusters.get(0)) + get_WSS(total_depth, max_depth, child_clusters.get(1));	}

		}

		return calculate_WSS(cluster);
	}

	public void elbow_method(int total_depth, Cluster cluster) {
		for (int i = 0; i < total_depth; i++) {
			double WSS = get_WSS(0,i, cluster);
			elbow_data.put(i, WSS);


		}
	}


	public void make_plot() {

		final ElbowPlotter demo = new ElbowPlotter("Elbow method", elbow_data );
		demo.pack();
		demo.setVisible(true);

	}
	
	public double get_cosine_similarity(double[] vec1, double[] vec2) {	
		return (dotProduct(vec1, vec2)/(getMagnitude(vec2) * getMagnitude(vec1)));
	}
	
	public String get_maximum_average_similarity(List<double[]> vector_list) {
		double[] max_vector = vector_list.get(0);
		double max_average_similarity = 0;
		for (double[] base_vector: vector_list) {
			double average_similarity = 0;
			int count = 0;
			for (double[] to_compare: vector_list) {
				if (!base_vector.equals(to_compare)){
					count += 1;
					average_similarity += get_cosine_similarity(base_vector, to_compare);

				}
			}
			average_similarity = average_similarity / count;
			if(average_similarity > max_average_similarity) {
				max_average_similarity = average_similarity;
				max_vector = base_vector;
			}
		}

		return wordvector_term.get(max_vector);
	}
	
	public void rename_subclusters(int cut_off_depth, int current_depth, Cluster cluster) {
		if (!cluster.isLeaf()) {
			List<double[]> vector_list = new ArrayList<double[]>();
			List<double[]> vectors_list = get_vector_list(cluster, vector_list);
			
			String new_name = get_maximum_average_similarity(vectors_list);
			cluster.setName(new_name);

			if (current_depth < cut_off_depth) {
				List<Cluster> child_clusters = cluster.getChildren();
				if (child_clusters.size() == 1) {
					rename_subclusters(cut_off_depth, current_depth + 1, child_clusters.get(0));
				}
				if (child_clusters.size() == 2) {
					rename_subclusters(cut_off_depth, current_depth + 1, child_clusters.get(0));
					rename_subclusters(cut_off_depth, current_depth + 1, child_clusters.get(1));
				}
			}
		}	
	}
	
	public void addToMap(String mapKey, String word_to_add) {
		List<String> itemsList = cluster_representation.get(mapKey);

		// if list does not exist create it
		if(itemsList == null)  {
			itemsList = new ArrayList<String>(); 
			itemsList.add(word_to_add);
			cluster_representation.put(mapKey, itemsList); 
		}     
		else 
		{ // add if item is not already in list
			if(!itemsList.contains(word_to_add)) 
				itemsList.add(word_to_add); 
		} 
	}
	
	public void leaf_node_handler(Cluster original_cluster, Cluster cluster) {
		if (!cluster.isLeaf()) {
			List<Cluster> child_clusters = cluster.getChildren();
			if (child_clusters.size() == 1) {
				leaf_node_handler(original_cluster, child_clusters.get(0));
			}
			if (child_clusters.size() == 2) {
				leaf_node_handler(original_cluster, child_clusters.get(0));
				leaf_node_handler(original_cluster, child_clusters.get(1));
			}

		}
		else {
			addToMap(original_cluster.getName(), cluster.getName());
		}
	}
	
	public void create_cluster_representation(Cluster cluster, int current_depth, int cut_off_depth) {
		if (!cluster.isLeaf()) {
			List<String> child_names = new ArrayList<String>();
			List<Cluster> child_clusters = cluster.getChildren();
			for (Cluster child: child_clusters) {
				child_names.add(child.getName());
			}
			cluster_representation.put(cluster.getName(), child_names);
			if (current_depth < cut_off_depth) {
				if (child_clusters.size() == 1) {
					create_cluster_representation(child_clusters.get(0), current_depth + 1, cut_off_depth);
				}
				if (child_clusters.size() == 2) {
					create_cluster_representation(child_clusters.get(0), current_depth + 1, cut_off_depth);
					create_cluster_representation(child_clusters.get(1), current_depth + 1, cut_off_depth);
				}
			}
			else {
				leaf_node_handler(cluster, cluster);
			}
		}
	}
	
	public static void main(String[] args) throws ClassNotFoundException, IOException {
	//WordvectorClustering test = new WordvectorClustering(Framework.OUTPUT_PATH + "word_vec_yelp_map");
	//clusterWords(getMap()); 
	//ClusteringAlgorithm alg = new DefaultClusteringAlgorithm(); 
	
			 
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
}
