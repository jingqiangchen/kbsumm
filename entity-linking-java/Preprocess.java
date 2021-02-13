package kbsumm;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.PrintWriter;
import java.rmi.RemoteException;
import java.rmi.registry.LocateRegistry;
import java.rmi.registry.Registry;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Scanner;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import edu.stanford.nlp.dcoref.CorefChain;
import edu.stanford.nlp.dcoref.CorefChain.CorefMention;
import edu.stanford.nlp.dcoref.CorefCoreAnnotations.CorefChainAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.NamedEntityTagAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations.TreeAnnotation;
import edu.stanford.nlp.trees.semgraph.SemanticGraph;
import edu.stanford.nlp.trees.semgraph.SemanticGraphCoreAnnotations.CollapsedCCProcessedDependenciesAnnotation;
import edu.stanford.nlp.util.CoreMap;
import javatools.parsers.Char;
import mpi.aida.data.Entity;
import mpi.aida.data.Mention;
import mpi.aidalight.rmi.AIDALightServer;
import mpi.aidalight.rmi.AIDALight_client;

public class Preprocess_NYT 
{
	public static String path_base = "/home/test/kbsumm";
	public static String path_corpus = path_base + "/data/nyt";
	public static String path_parser = "/home/test/eclipse/libs/stanford-parser/stanford-parser.jar";
	public static String path_tokens = path_corpus + "/tokens";
	public static String path_stories = path_corpus + "/stories";
	public static String path_stories_t = path_corpus + "/stories-t";
	public static String path_stories_ts = path_corpus + "/stories-ts";
	public static String path_mentions = path_corpus + "/mentions";
	public static String path_mentions_ex = path_corpus + "/mentions-ex";
	public static String path_entities = path_corpus + "/entities";
	public static String path_summaries = path_corpus + "/summaries";
	public static String path_summaries_starts = path_corpus + "/summary-starts";
	
	public static String path_valid_file = path_corpus+"/valid";
	public static String path_train_file = path_corpus+"/trains";
	public static String path_dev_file = path_corpus+"/devs";
	public static String path_test_file = path_corpus+"/tests";

	private void tokenizeAndSplit() throws Exception
	{
		Properties props = new Properties();
		props.setProperty("annotators", "tokenize, ssplit"); //, pos, lemma, ner, parse, dcoref
		StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
		
		File dir=new File(path_stories_ts);
		if(!dir.exists()) dir.mkdirs();
		
		Scanner sc=new Scanner(new FileInputStream(path_valid_file));
		PrintWriter pw=new PrintWriter(new FileOutputStream(path_summaries_starts));
		while(sc.hasNextLine())
		{
			String file_name=sc.nextLine();
			
			Scanner sc2=new Scanner(new FileInputStream(path_stories+"/"+file_name));
			PrintWriter pw2=new PrintWriter(new FileOutputStream(path_stories_ts+"/"+file_name));
			
			String sign="";
			String story="", text="", summary="";
			String headline="", byline="", full_text="";
			while(sc2.hasNextLine())
			{
				String line=sc2.nextLine().trim();
				if(line.equals("[HEADLINE]"))
					sign="[HEADLINE]";
				else if(line.equals("[BYLINE]"))
					sign="[BYLINE]";
				else if(line.equals("[ABSTRACT]"))
					sign="[ABSTRACT]";
				else if(line.equals("[FULL_TEXT]"))
					sign="[FULL_TEXT]";
				else if(!line.equals(""))
				{
					char lastChar=line.charAt(line.length()-1);
					if(lastChar>='a' && lastChar<='z' || lastChar>='A' && lastChar<='Z')
						line+=".";
					if(sign.equals("[HEADLINE]"))
						headline+=" "+line;
					else if(sign.equals("[BYLINE]"))
						byline+=" "+line;
					else if(sign.equals("[FULL_TEXT]"))
						full_text+=" "+line;
					else if(sign.equals("[ABSTRACT]"))
						summary+=" "+line;
				}
			}
			sc2.close();
			
			headline=headline.trim();
			byline=byline.trim();
			full_text=full_text.trim();
			summary=summary.trim();
			text=headline+" "+byline+" "+full_text;
			
			summary=pre_summary2(summary.trim()).trim();
			
			Annotation document = new Annotation(text);
			pipeline.annotate(document);
			List<CoreMap> sentences = document.get(SentencesAnnotation.class);
			for(CoreMap sentence: sentences) 
			{
			  for (CoreLabel token: sentence.get(TokensAnnotation.class)) 
			  {
			    String word = token.get(TextAnnotation.class);
			    pw2.print(word+" ");
			  }
			  pw2.println();
			}
			
			pw.println(file_name+"\t"+(sentences.size()+1));
			
			document = new Annotation(summary);
			pipeline.annotate(document);
			sentences = document.get(SentencesAnnotation.class);
			for(CoreMap sentence: sentences) 
			{
			  for (CoreLabel token: sentence.get(TokensAnnotation.class)) 
			  {
			    String word = token.get(TextAnnotation.class);
			    pw2.print(word+" ");
			  }
			  pw2.println();
			}
			
			pw2.close();
		}
		pw.close();
		sc.close();
	}
	
	public void ner() throws Exception
	{
		Properties props = new Properties();
		props.setProperty("annotators", "tokenize, ssplit, pos, lemma, ner, parse, dcoref"); //
		StanfordCoreNLP pipeline = new StanfordCoreNLP(props);

		File dir=new File(path_mentions_ex);
		if(!dir.exists()) dir.mkdirs();
		
		ArrayList<String> entityList=new ArrayList<String>();
		Hashtable<String,ArrayList<EntityPosition> > entities=new Hashtable<String,ArrayList<EntityPosition> >();
		
		PrintWriter pwerror=new PrintWriter(new FileOutputStream("ner-error"), true);
		Scanner sc=new Scanner(new FileInputStream(path_test_file));
		int count=0;
		while(sc.hasNextLine())
		{
			String file_name=sc.nextLine();
			count++;
			if(count<0)continue;
			if(count==60000)break;
			File output=new File(path_mentions_ex+"/"+file_name);
			if(output.exists()) 
			{
				System.out.println("exists:"+file_name);
				continue;
			}
			if(file_name.equals("1174129") || file_name.equals("1245697") || file_name.equals("1267993"))
				continue;
			
			Scanner sc2=new Scanner(new FileInputStream(path_stories_ts+"/"+file_name));
			String text="";
			while(sc2.hasNextLine())
				text+=sc2.nextLine()+" ";
			text=text.trim();
			sc2.close();
			
			entityList.clear();
			entities.clear();
			try
			{
				ner(pipeline,text,entityList,entities);
			}
			catch(OutOfMemoryError e)
			{
				pwerror.println(file_name);
				System.out.println("error:"+file_name);
				continue;
			}
			PrintWriter pw2=new PrintWriter(new FileOutputStream(path_mentions_ex+"/"+file_name));
			for(int i=0;i<entityList.size();i++)
			{
				String entity=entityList.get(i);
				ArrayList<EntityPosition> eplist=entities.get(entity);
				for(EntityPosition ep:eplist)
				{
					pw2.println((i+1)+"\t"+entity+"\t"+ep.sentPos+"\t"+ep.start+"\t"+ep.end);
				}
			}
			pw2.close();
			System.out.println(file_name+"\t"+count);
		}
		sc.close();
		pwerror.close();
	}
	
	public void nerTest()
	{
		Properties props = new Properties();
		props.setProperty("annotators", "tokenize, ssplit, pos, lemma, ner, parse, dcoref"); //
		StanfordCoreNLP pipeline = new StanfordCoreNLP(props);

		String text = "The Nutty Professor. " 
				+ "Eddie Murphy, Jada Pinkett  Directed by Tom Shadyac  PG-13 95 minutes. "
				+ "Mr. Murphy plays Sherman Klump, the bumbling, kindhearted chemistry professor "
				+ "created by Jerry Lewis in the 1963 film of the same name. But in this remake, "
				+ "Sherman is also horrifically obese, thanks to makeup and special effects. "
				+ "Trying to win the heart of a beautiful young graduate student (Ms. Pinkett), "
				+ "he makes himself the experimental subject for his latest research project: "
				+ "a weight-loss formula that restructures DNA. For limited periods, the drug slims Sherman down, "
				+ "but it also raises his testosterone levels, unleashing a leering, swearing, sex-obsessed alter ego. "
				+ "VIOLENCE A few punches and one cartoonlike fight.  SEX Mostly verbal references.  PROFANITY Frequent and extremely crass. " 
				+ "For Which Children? " 
				+ "AGES 6-9 Laboratory hamsters' antics and a running joke about flatulence appeal directly "
				+ "to the sense of humor of elementary school students, but don't take them unless you want them imitating "
				+ "Mr. Murphy's vocabulary and sound effects at the dinner table.  "
				+ "AGES 10-13 If being grossed out is their idea of entertainment, they'll love it.  "
				+ "AGES 14 and up Only for die-hard Murphy fans. " 
				+ "LAUREL GRAEBER. TAKING THE CHILDREN.";
		ArrayList<String> entityList=new ArrayList<String>();
		Hashtable<String,ArrayList<EntityPosition> > entities=new Hashtable<String,ArrayList<EntityPosition> >();
		ner(pipeline,text,entityList,entities);
		for(String entity:entityList)
		{
			System.out.println(entity);
			ArrayList<EntityPosition> eplist=entities.get(entity);
			for(EntityPosition ep:eplist)
			{
				System.out.println(ep.sentPos+"-"+ep.start+"-"+ep.end+"-"+ep.start2+"-"+ep.end2);
			}
		}
	}
	
	public void ner(StanfordCoreNLP pipeline,String text,ArrayList<String> entityList,Hashtable<String,ArrayList<EntityPosition> > entities)
	{
		Annotation document = new Annotation(text);
		pipeline.annotate(document);
		
		List<CoreMap> sentences = document.get(SentencesAnnotation.class);
		//Hashtable<String,ArrayList<EntityPosition> > entities=new Hashtable<String,ArrayList<EntityPosition> >();
		//ArrayList<String> entityList=new ArrayList<String>();
		String entity="";
		int totalTokens=0, start=0, end=0;
		Hashtable<String,Boolean> emap=new Hashtable<String,Boolean>();
		ArrayList<Integer> sentTokenIndices=new ArrayList<Integer>();
		for(int sentIndex=0;sentIndex<sentences.size();sentIndex++) 
		{
			CoreMap sentence = sentences.get(sentIndex);
			List<CoreLabel> tokens = sentence.get(TokensAnnotation.class);
			//System.out.println(sentence.toString());
			sentTokenIndices.add(totalTokens+1);
			for (int tokenIndex=0;tokenIndex<tokens.size();tokenIndex++) 
			{
				CoreLabel token = tokens.get(tokenIndex);
				String word = token.get(TextAnnotation.class);
				String pos = token.get(PartOfSpeechAnnotation.class);
				String ne = token.get(NamedEntityTagAnnotation.class);
				if (ne.equals("PERSON") || ne.equals("LOCATION") || ne.equals("ORGANIZATION"))
				{
					entity=token.value();
					start=end=totalTokens+tokenIndex+1;
					String word2, pos2, ne2;
					for(tokenIndex++;tokenIndex<tokens.size();tokenIndex++)
					{
						token = tokens.get(tokenIndex);
						word2 = token.get(TextAnnotation.class);
						pos2 = token.get(PartOfSpeechAnnotation.class);
						ne2 = token.get(NamedEntityTagAnnotation.class);
						if(ne.equals(ne2))
						{
							entity+=" "+token.value();
							end++;
						}
						else break;
					}
					tokenIndex--;
					ArrayList<EntityPosition> eplist=entities.get(entity);
					if(eplist==null)
					{
						eplist=new ArrayList<EntityPosition>();
						entities.put(entity, eplist);
						entityList.add(entity);
					}
					eplist.add(new EntityPosition(sentIndex+1,start,end,tokenIndex,tokenIndex+1+end-start));
					emap.put(entity+"-"+(sentIndex+1)+"-"+(tokenIndex)+"-"+(tokenIndex+1+end-start), true);
					//System.out.println(entity+"-"+(sentIndex+1)+"-"+(tokenIndex+1)+"-"+(tokenIndex+1+end-start)+"-"+start+"-"+end);
				}
			}
			totalTokens+=tokens.size();
			Tree tree = sentence.get(TreeAnnotation.class);
			
			//SemanticGraph dependencies = sentence.get(CollapsedCCProcessedDependenciesAnnotation.class);
		}
		/*
		for(String key:entityList)
		{
			List<EntityPosition> list=entities.get(key);
			System.out.println(key);
			for(EntityPosition ep:list)
			{
				System.out.println(ep.sentPos+"\t"+ep.start+"\t"+ep.end);
			}
		}*/
		ArrayList<ArrayList<String> > mentions=new ArrayList<ArrayList<String> >();
		
		Map<Integer, CorefChain> graph = document.get(CorefChainAnnotation.class);
		Iterator iter = graph.keySet().iterator();
		while(iter.hasNext())
		{
			Integer key = (Integer)iter.next();
			CorefChain cc = graph.get(key);
			boolean sign=false;
			List<CorefMention> ll = cc.getMentionsInTextualOrder();
			for(CorefMention m:ll)
			{
				//System.out.println(m.mentionSpan+"-"+m.sentNum+"-"+m.startIndex+"-"+m.endIndex);
				if(emap.get(m.mentionSpan+"-"+m.sentNum+"-"+m.startIndex+"-"+m.endIndex)!=null)
				{
					sign=true;
					break;
				}
			}
			if(sign)
			{
				for(int i=0;i<ll.size();i++)
				{
					CorefMention m=ll.get(i);
					if(m.mentionSpan.split(" ").length>4) ll.remove(i);
				}
				if(ll.size()>0)
				{
					ArrayList<String> aa=new ArrayList<String>();
					mentions.add(aa);
					for(int i=0;i<ll.size();i++)
					{
						CorefMention m=ll.get(i);
						ArrayList<EntityPosition> eplist=entities.get(m.mentionSpan);
						if(eplist==null)
						{
							eplist=new ArrayList<EntityPosition>();
							entities.put(m.mentionSpan, eplist);
							int j;
							for(j=0;j<entityList.size();j++)
								if(entities.get(entityList.get(j)).get(0).start>sentTokenIndices.get(m.sentNum-1)+m.startIndex-1)
									break;
							entityList.add(j, m.mentionSpan);
						}
						if(emap.get(m.mentionSpan+"-"+m.sentNum+"-"+m.startIndex+"-"+m.endIndex)==null)
						{
							eplist.add(new EntityPosition(m.sentNum,sentTokenIndices.get(m.sentNum-1)+m.startIndex-1,
									sentTokenIndices.get(m.sentNum-1)+m.endIndex-2,m.startIndex,m.endIndex));
							emap.put(m.mentionSpan+"-"+m.sentNum+"-"+m.startIndex+"-"+m.endIndex, true);
						}
						if(aa.indexOf(m.mentionSpan)==-1)
							aa.add(m.mentionSpan);
						//System.out.println(m.mentionSpan+" | "+m.mentionID+" | "+m.startIndex+" | "+m.endIndex+" | "+m.headIndex+" | "+m.sentNum);
					}
				}
				
				//System.out.println();
			}
		}
		
		for(ArrayList<String> a:mentions)
		{
			String mention0=a.get(0);
			String key=mention0;
			ArrayList<EntityPosition> b=entities.get(mention0);
			for(int i=1;i<a.size()&&b!=null;i++)
			{
				String mention=a.get(i);
				if(entities.get(mention)==null) continue;
				b.addAll(entities.get(mention));
				entities.remove(mention);
				entityList.remove(mention);
				if(key.length()<mention.length()) key=mention;
			}
			
			int keyIndex=entityList.indexOf(mention0);
			if(keyIndex==-1) continue;
			entityList.remove(mention0);
			entityList.add(keyIndex,key);
			entities.remove(mention0);
			entities.put(key, b);
		}
		
		for(String e:entityList)
		{
			ArrayList<EntityPosition> eplist=entities.get(e);
			if(eplist.size()>1)
			{
				eplist.sort(new Comparator<EntityPosition>(){
	                public int compare(EntityPosition a0, EntityPosition a1) {
	                	return a0.start-a1.start;
	                     //return arg0.getId().compareTo(arg1.getId());//这是顺序
	                }               
	           });  
			}
		}
	}
	
	private String pre_summary(String s)
	{
		s=s.replaceAll("\\([s|S]\\)", "").trim();
		s=s.replaceAll("\\([m|M]\\)", "").trim();
		s=s.replaceAll("photo$", "");
		s=s.replaceAll("graph$", "");
		s=s.replaceAll("chart$", "");
		s=s.replaceAll("map$", "");
		s=s.replaceAll("table$", "");
		s=s.replaceAll("drawing$", "");
		
		//s=s.replaceAll("photo", ": ");
		//s=s.replaceAll("graph", ": ");
		//s=s.replaceAll(": chart", ": ");
		//s=s.replaceAll(": map", ": ");
		//s=s.replaceAll(": table", ": ");
		//s=s.replaceAll(": drawing", ": ");
		
		//s=s.replaceAll("photo;", ";");
		//s=s.replaceAll("graph;", ";");
		//s=s.replaceAll("chart;", ";");
		//s=s.replaceAll("map;", ";");
		//s=s.replaceAll("table;", ";");
		//s=s.replaceAll("drawing;", ";");
		return s;
	}
	
	private String pre_summary2(String text)
	{
		String expression1="\\([m|s]\\)";
		String expression2=";\\s*(?:photo|graph|chart|map|table|drawing)\\s*;?\\s*$";
		Pattern p1 = Pattern.compile(expression1, Pattern.CASE_INSENSITIVE); // 正则表达式
		Matcher m1 = p1.matcher(text);
		text = m1.replaceAll("");
		
		Pattern p2 = Pattern.compile(expression2, Pattern.CASE_INSENSITIVE); // 正则表达式
		Matcher m2 = p2.matcher(text); // 操作的字符串
		text = m2.replaceAll(";"); //替换后的字符串 
		text = text.replaceAll(";", ".").trim();
		
		int len = text.length();
		char lastChar = text.charAt(len-1);
		if(lastChar>='a'&&lastChar<='z'||lastChar>='A'&&lastChar<='Z'||lastChar>='0'&&lastChar<='9')
			text += ".";
		return text;
	}
	
	private void test()
	{
		String expression1="\\([m|s]\\)";
		String expression2=";\\s*(?:photo|graph|chart|map|table|drawing)\\s*;?\\s*$";
		String text="Standard & Poor's 500-stock index falls 1.86 points to 1,531.05; Dow Jones industrial average drops 26.50 points to 13,612.98; "
				+ "Nasdaq composite index slips 0.11 point to 2,626.60; "
				+ "Russell 2000 index drops 0.2 percent, to 846.28; "
				+ "price and yield of 10-year US Treasury note noted; "
				+ "crude oil for July delivery rises $1.09 to $69.09 in New York trading after Nigerian unions plan strike this week, "
				+ "threatening supplies from Africa's biggest producer; graph (M)";
		text = "Sen Barack Obama says his campaign 'made mistake' when it distributed document criticizing support "
				+ "Sen Hillary Rodham Clinton has received from Indian-Americans and companies that do business in India; "
				+ "says he did not know about memo, but considers himself responsible for mistake (S)";
		Pattern p1 = Pattern.compile(expression1, Pattern.CASE_INSENSITIVE); // 正则表达式
		Matcher m1 = p1.matcher(text);
		text = m1.replaceAll("");
		
		Pattern p2 = Pattern.compile(expression2, Pattern.CASE_INSENSITIVE); // 正则表达式
		Matcher m2 = p2.matcher(text); // 操作的字符串
		text = m2.replaceAll(";").replaceAll(";", ".").trim(); //替换后的字符串 
		
		int len = text.length();
		char lastChar = text.charAt(len-1);
		if(lastChar>='a'&&lastChar<='z'||lastChar>='A'&&lastChar<='Z'||lastChar>='0'&&lastChar<='9')
			text += ".";
		
		System.out.println(text);
	}

	
	class EntityPosition
	{
		int sentPos;
		int start,end;
		int start2,end2;
		public EntityPosition(int sentPos,int start,int end,int start2,int end2)
		{
			this.sentPos=sentPos;
			this.start=start;
			this.end=end;
			this.start2=start2;
			this.end2=end2;
		}
	}
	
	public static void main(String args[]) throws Exception 
	{
		Preprocess_NYT pp=new Preprocess_NYT();
		//pp.tokenizeAndSplit();
		pp.ner();
		//pp.nerTest();
		//pp.test();
	}
}










