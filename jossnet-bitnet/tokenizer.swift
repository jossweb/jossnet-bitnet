//
//  tokenizer.swift
//  jossnet-bitnet
//
//  Created by FIGUEIRAS Jossua on 03/02/2026.
//
import Foundation

struct TokenizerFile: Decodable {
    let model: TokenizerModel
}

struct TokenizerModel: Decodable {
    let vocab: [String: Int]
}

func getId(word:String)->UInt32?{
    let url = URL(fileURLWithPath: "/Users/jossua/Documents/jossnet-bitnet/bitnet-b1.58-2B-4T/tokenizer.json")
        
        do {
            let data = try Data(contentsOf: url)
            
            let tokenizerData = try JSONDecoder().decode(TokenizerFile.self, from: data)
            
            let vocab = tokenizerData.model.vocab
            
            
            if let id = vocab["Ġ" + word] {
                return UInt32(id)
            }
            else if let id = vocab[word] {
                return UInt32(id)
            }
            else {
                return nil
            }
        } catch {
            print("Error: \(error)")
            return nil
        }
}
func formatString(str:String)->[UInt32]{
    let pattern = #"(\w+|[^\w\s]+)"#
    var result : [UInt32] = []
    do{
        let regex = try NSRegularExpression(pattern: pattern);
        let nsString = str as NSString
        let results = regex.matches(in: str, range: NSRange(location: 0, length: nsString.length))
        
        for (_, match) in results.enumerated() {
            let word = nsString.substring(with: match.range)
                print("word : \(word)")
            if let id = getId(word: word) {
                result.append(id)
            }else{
                // TODO :
                // Check if a unknow token exist in tokenizer
                //result.append(0)
            }
        }
    } catch {
        print("ERROR REGEX : \(error)")
    }
    return result
}
