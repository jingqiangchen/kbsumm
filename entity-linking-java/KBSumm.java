package kbsumm;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintWriter;
import java.rmi.RemoteException;
import java.rmi.registry.LocateRegistry;
import java.rmi.registry.Registry;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

import javatools.parsers.Char;
import mpi.aida.data.Entity;
import mpi.aida.data.Mention;
import mpi.aidalight.rmi.AIDALightServer;
import mpi.aidalight.rmi.AIDALight_client;

public class KBSumm 
{
	public static String path_base = "/home/test/kbsumm";
	public static String path_corpus = path_base + "/data/nyt50";
	public static String path_parser = "/home/test/eclipse/libs/stanford-parser/stanford-parser.jar";
	public static String path_tokens = path_corpus + "/tokens";
	public static String path_stories = path_corpus + "/stories";
	public static String path_stories_t = path_corpus + "/stories-t";
	public static String path_stories_ts = path_corpus + "/texts-ts";
	public static String path_mentions = path_corpus + "/mentions";
	public static String path_mentions_ex = path_corpus + "/mentions-ex";
	public static String path_entities = path_corpus + "/entities";
	public static String path_summaries = path_corpus + "/summaries";
	
	public static String path_train_file = path_corpus+"/trains";
	public static String path_dev_file = path_corpus+"/devs";
	public static String path_test_file = path_corpus+"/tests";

	public static Map<Mention, Entity> disambiguate(String text, List<Mention> mentions, String host) throws RemoteException 
	{
		if (host == null)
			host = "localhost"; // default
		// set up server
		AIDALightServer server = null;
		try 
		{
			Registry registry = LocateRegistry.getRegistry(host, 52365);
			server = (AIDALightServer) registry.lookup("NEDServer_" + host);
		} 
		catch (Exception e) 
		{
			e.printStackTrace();
		}

		String command = "fullSettings"; // = key-words + 2-phase mapping + domain

		Map<Mention, Entity> annotations = server.disambiguate(text, mentions, command);

		// do whatever here...
		/*for (Mention mention : annotations.keySet()) 
		{
			String wikipediaEntityId = "http://en.wikipedia.org/wiki/" + annotations.get(mention).getName();
			System.out.println(mention.getMention() + "\t" + wikipediaEntityId + "\t" + mention.getStartToken() + "\t"
					+ mention.getEndToken() + "\t" + mention.getSentenceId() + "\t" + mention.getStartStanford() + "\t"
					+ mention.getEndStanford());
		}
		*/
		return annotations;
	}
	
	public static void entity_linking() throws Exception
	{
		String[] fileNames = {path_train_file, path_dev_file, path_test_file};
		//String[] fileNames = {path_train_file};
		//path_train_file=path_corpus+"/test-test-test";
		//String[] fileNames = {path_train_file};
		PrintWriter pwNot=new PrintWriter(new FileOutputStream("notexist"),true);
		int count=0;
		for(String fileName:fileNames)
		{
			Scanner scFile=new Scanner(new FileInputStream(fileName));
			while(scFile.hasNextLine())
			{
				
				String file_name=scFile.nextLine();
				System.out.println(++count + ":" + file_name);
				//if(count<160300)
				//	continue;
				Scanner scStory=new Scanner(new FileInputStream(path_stories_ts+"/"+file_name));
				String story="";
				while(scStory.hasNextLine())
					story+=scStory.nextLine()+"\n";
				scStory.close();
				
				try
				{
					HashMap<String,String> mention2Ids=new HashMap<String,String>();
					ArrayList<Mention> mentions=new ArrayList<Mention>();
					Scanner scEntity=new Scanner(new FileInputStream(path_mentions_ex+"/"+file_name));
					while(scEntity.hasNextLine())
					{
						String line=scEntity.nextLine();
						String [] items=line.split("\t");
						Mention m=new Mention(items[1], Integer.parseInt(items[3])-1, Integer.parseInt(items[4])-1, 
								Integer.parseInt(items[3])-1, Integer.parseInt(items[4])-1, Integer.parseInt(items[2])-1);
						mentions.add(m);
						String value=mention2Ids.get(items[1]);
						if (value==null)
							mention2Ids.put(items[1], "+"+items[0]+"+");
						else if (value.indexOf("+"+items[0]+"+")==-1)
							mention2Ids.put(items[1], value+items[0]+"+");
					}
					scEntity.close();
					
					Map<Mention, Entity> map=disambiguate(story, mentions, null);
					PrintWriter pw = new PrintWriter(new FileOutputStream(path_entities+"/"+file_name));
					for (Mention mention : map.keySet()) 
					{
						String entityName = map.get(mention).getName();
						int entityId = map.get(mention).getId();
						//System.out.println(mention2Ids.get(mention.getMention())+"\t"+mention.getMention());
						pw.println(mention2Ids.get(mention.getMention())+"\t"+mention.getMention() + "\t" + entityName+"\t"+entityId);
					}
					pw.close();
				}
				catch(FileNotFoundException e)
				{
					pwNot.println(file_name);
					System.out.println(file_name+" not exist.");
				}
			}
			scFile.close();
		}
		
		pwNot.close();
	}
	
	public static void test() throws Exception
	{
		String story="Authorities say he then stole a pistol in eastern British Columbia and took a plane from a hangar in Idaho , where investigators found bare footprints on the floor and wall . That plane crashed near Granite Falls , Wash. , after it ran out of fuel , prosecutors said . " + 
				"He made his way to Oregon in a 32 - foot boat stolen in southwestern Washington - stopping first to leave $ 100 at an animal shelter in Raymond , Wash . From Oregon , authorities said , Harris - Moore travelled across the United States until he made it to the Bahamas " + 
				"In all , Harris - Moore is suspected of more than 100 crimes across nine states . " + 
				"Harris - Moore 's lawyer , John Henry Browne , said his client has already agreed that he does not want to benefit from his past behaviour . " + 
				"' Everything is already agreed to . Colton does not and did not want a dime and thinks it 's wrong to benefit from this , ' Browne told the Seattle Times . " + 
				"Captured : A sign at a real estate office in Washington shows support for the capture of Harris - Moore " + 
				"Colton Harris - Moore is suspected of more than 100 crimes across nine states , in an alleged spree from April 2008 until July 2010 . ";
		Map<Mention, Entity> map=disambiguate(story, null, null);
		Iterator iter=map.keySet().iterator();
		while(iter.hasNext())
		{
			Mention key=(Mention)iter.next();		
			Entity entity=map.get(key);
			System.out.println(key.getMention()+"\t"+ entity.toString()+"\t"+ key.getSentenceId()
								+"\t"+ key.getStartToken()+"\t"+key.getEndToken()
								+"\t"+key.getStartStanford()+"\t"+key.getEndStanford());
		}
	}

	public static void main(String args[]) throws Exception 
	{
		// Mention(String mention, int startToken, int endToken, int startStanford, int
		// endStanford, int sentenceId)
		// Granite_Falls\u0027s_Minnesota
		//AIDALight_client.disambiguate("", null, null);
		
		entity_linking();
		
		//test();
		
		//String s="Granite_Falls\\u0027s_Minnesota";
		//String s="现，Python3以后";
		//System.out.println(Char.encodeBackslash(s));
	}
}










