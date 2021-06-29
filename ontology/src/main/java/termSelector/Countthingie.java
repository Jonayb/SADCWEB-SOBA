package termSelector;

public class Countthingie {

	public static void main(String[] args) {
		String myString = "{35=[38, 41], 48=[6, 20], 38=[45, clstr#41, 47, 13, 38], 6=[6, 17], 18=[18, 31], 41=[clstr#42, clstr#45, 4, 41, 1, 35], 31=[54, 18], 20=[30, clstr#40, 37, 51, 0, 39, 3, 16, 12, 26, 27, 11, 23, 15, 34, 8, 5, 46, 36, 21, 20, 44, 33, 32, 48, 49, 43, 52, 7, 14, 24, 29, 9, 50, 22, 10, 28, 53, 25, 40, 2, 42]}";
	boolean everything = true; 
	for(int i = 0; i < 55; i++) {
		String p = Integer.toString(i); 
		if(!myString.contains(p))
			everything = false; 
		System.out.println(p);
	}
	
	System.out.println(everything); 
}
}