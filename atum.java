/*****
Author: Sikder Tahsin Al Amin
Problem: Write a function in Java that takes as a parameter a string and returns a list of subsequences (from left to right) that start with "ATG" and end with either "TAA", "TAG" or "TGA".
Example: 
Input: "GCATGCCGGTTACCTAAGGATGGGTTCAAAATAGCGG"
Output: ["ATGCCGGTTACCTAA", "ATGGGTTCAAAATAG"]
Command to Compile and Run the program:
    javac atum.java
    java atum
*/


import java.util.ArrayList;

class helper{
    static ArrayList<String> subsequence(String s)
    {
        ArrayList<String> substrings = new ArrayList<String>(); //create a new arraylist object
        
        for (int i=0;i<s.length()-2;i++){
            String substr1 = s.substring(i,i+3);
            if (substr1.equals("ATG")){
                String sequence = substr1; //actual subsequence
                int j = i+substr1.length();
                
                while (j<s.length()-2){
                    String substr2 = s.substring(j,j+3); //finding the tail
                    if (substr2.equals("TAA") ||substr2.equals("TAG")|| substr2.equals("TGA")){
                        sequence += substr2;
                        substrings.add(sequence);
                        i = j+3;
                        break;
                    }
                    sequence += s.charAt(j);
                    j = j+1;
                }
            }
        }
        return substrings ;
    }
}


public class atum{
    public static void main(String args[]){
        String a = "GCATGCCGGTTACCTAAGGATGGGTTCAAAATAGCGG"; //input string
        ArrayList<String> res = new ArrayList<String>();
        res = helper.subsequence(a);
        System.out.println(res);

    }
}
