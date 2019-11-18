import os
import lyricsgenius
import utils.utilities as U

# Sign up for a free account at Genius.com to access the API - http://genius.com/api-clients
client_access_token = '1iI3Y561yvR9KYXagktbvUR1GXNjQDCuIYh4tk1YjJKbaQW2N3Ahq097AGTznLaI'
# A small library based on the above API - https://github.com/johnwmillr/LyricsGenius
genius = lyricsgenius.Genius(client_access_token)

# genius.verbose = False # Turn off status messages
genius.remove_section_headers = True  # Remove section headers (e.g. [Chorus]) from lyrics when searching
genius.excluded_terms = ["(Remix)", "(Live)", "(Radio Edit)"]  # Exclude songs with these words in their title

POP_ARTISTS = {
    "genre": "Pop",
    "artists": [
        "Justin Bieber", "Adele", "Drake", "The Weeknd", "Taylor Swift", "Selena Gomez",
        "One Direction", "Shawn Mendes", "Meghan Trainor", "Ellie Goulding", "Rachel Platten",
        "Charlie Puth", "Demi Lovato", "Ariana Grande", "Ed Sheeran", "Alessia Cara",
        "Sam Smith", "Coldplay", "Maroon 5", "Beyoncé", "Nicki Minaj", "Justin Timberlake", "John Legend",
        "Lana Del Rey", "Rihanna", "Lady Gaga", "Katy Perry", "Miley Cyrus", "Sia", "Calvin Harris",
        "Robin Thicke", "Camila Cabello", "Michael Jackson", "Elton John", "Whitney Houston",
        "Madonna", "Bruno Mars", "Mariah Carey", "Harry Styles", "Marshmello",
        "P!nk", "Imagine Dragons", "Christina Aguilera", "Prince", "Stevie Wonder", "Normani", "Britney Spears",
        "Nickelback", "Rascal Flatts", "Ne-Yo", "Sean Paul", "The Pussycat Dolls", "James Blunt", "The Fray",
        "Kelly Clarkson", "The Black Eyed Peas", "Natasha Bedingfield", "Fergie",
        "Jamie Foxx", "Shakira", "Nelly", "T-Pain", "Kesha", "Snow Patrol", "Daniel Powter",
        "Janet Jackson", "Celine Dion", "R. Kelly", "Monica", "Toni Braxton", "Brandy", "Rod Stewart",
        "Jennifer Lopez", "Spice Girls", "Seal", "Backstreet Boys", "George Michael", "Amy Grant",
        "Ace of Base", "Soul Decision", "5 Seconds of Summer", "M2M", "Ricky Martin", "Steps",
        "Sugar Ray", "Natalia Oreiro"
    ]
}

RAP_ARTISTS = {
    "genre": "Rap",
    "artists": [
        "Eminem", "Kendrick Lamar", "Big Shaq", "Cardi B", "Travis Scott", "Logic",
        "Kanye West", "Lil Uzi Vert", "Fetty Wap", "Post Malone", "JAY-Z", "Lil Wayne", "Snoop Dogg",
        "2Pac", "J. Cole", "50 Cent", "T.I.", "Dr. Dre", "Migos", "Future", "OutKast", "Busta Rhymes",
        "A$AP Rocky", "Ice Cube", "Rick Ross", "2 Chainz", "Wu-Tang Clan", "Juice Wrld", "Young Thug",
        "Big Sean", "Ghostface Killah", "Donald Glover & Danny Pudi", "Jeezy", "DMX", "Chance the Rapper", "Rakim",
        "LL Cool J", "Wiz Khalifa", "Chris Brown", "N.W.A", "Scarface", "XXXTENTACION", "The Notorious B.I.G.", "Nas",
        "André 3000", "Childish Gambino", "Nate Dogg", "Joyner Lucas", "The Game",
        "Ludacris", "Kid Cudi", "Redman", "Lauryn Hill", "KRS-One", "Big Pun", "Pusha T", "Hopsin", "ScHoolboy Q",
        "Meek Mill", "Raekwon", "Xzibit", "Slick Rick", "Denzel Curry", "MF DOOM", "Missy Elliott",
        "Big Daddy Kane", "Lupe Fiasco", "Gucci Mane", "Warren G", "RZA", "Q-Tip", "GZA", "B.o.B",
        "Twista", "21 Savage", "Big Boi", "Lil Dicky", "Prodigy of Mobb Deep", "NF", "Proof", "Talib Kweli",
        "Ski Mask the Slump God", "Obie Trice", "G-Eazy", "Yelawolf", "Earl Sweatshirt", "Chuck D", "MC Ren", "YG",
        "DaBaby"
    ]
}

COUNTRY_ARTISTS = {
    "genre": "Country",
    "artists": [
        "Johnny Cash", "George Strait", "Dolly Parton", "Willie Nelson", "Garth Brooks", "Waylon Jennings",
        "Alan Jackson", "Merle Haggard", "George Jones", "Loretta Lynn", "Reba McEntire", "Carrie Underwood",
        "Shania Twain", "Patsy Cline", "Hank Williams", "Kenny Rogers", "Miranda Lambert",
        "Brad Paisley", "Vince Gill", "Dixie Chicks", "Buck Owens", "Tim McGraw", "Randy Travis", "Charley Pride",
        "Faith Hill", "Conway Twitty", "Tammy Wynette", "Emmylou Harris", "Kris Kristofferson", "Ray Price",
        "Keith Urban", "Glen Campbell", "Alabama", "Toby Keith", "Blake Shelton", "Linda Ronstadt",
        "Brooks & Dunn", "Dwight Yoakam", "John Denver", "Alison Krauss", "Jerry Jeff Walker", "Chris LeDoux",
        "Lucinda Williams", "Crystal Gayle", "The Judds", "Jamey Johnson", "Bill Anderson", "Eric Church",
        "Steve Earle", "Townes Van Zandt", "Lynn Anderson", "The Statler Brothers", "Rosanne Cash", "Patty Loveless",
        "Marty Stuart", "Asleep at the Wheel", "Lee Ann Womack", "Merle Travis", "Jessi Colter",
        "Vern Gosdin", "Connie Smith", "Guy Clark", "Tanya Tucker", "Roy Acuff", "Billy Joe Shaver",
        "The Carter Family", "Jimmie Rodgers", "Eddy Arnold", "Roger Miller",
        "Lefty Frizzell", "Ernest Tubb", "The Louvin Brothers", "Tom T. Hall", "Bill Monroe",
        "Flatt & Scruggs", "John Prine", "Cody Johnson", "Chris Stapleton", "Luke Combs", "Midland",
        "Cole Swindell", "Brantley Gilbert", "Chris Janson", "LANco", "Jon Pardi", "Dylan Scott", "Brett Young",
        "Brothers Osborne", "Old Dominion", "Dustin Lynch", "Michael Ray", "Aaron Watson", "Chris Lane", "Kane Brown"
    ]
}

ROCK_ARTISTS = {
    "genre": "Rock",
    "artists": [
        "The Beatles", "The Rolling Stones", "Led Zeppelin", "Pink Floyd", "Queen", "David Bowie", "Elvis Presley",
        "AC/DC", "Guns N Roses", "Fleetwood Mac", "Metallica", "Bob Dylan", "Aerosmith", "Black Sabbath", "U2",
        "Bruce Springsteen", "The Who", "Paul McCartney", "The Doors", "Chuck Berry", "Ozzy Osbourne",
        "The Beach Boys", "Nirvana", "Eagles", "Van Halen", "Eric Clapton",
        "The Police", "Lynyrd Skynyrd", "Rush", "Deep Purple", "Red Hot Chili Peppers", "Pearl Jam",
        "Dire Straits", "Journey", "The Clash", "Radiohead", "The Kinks", "ZZ Top", "Foo Fighters", "Green Day",
        "Iron Maiden", "Ramones", "Genesis", "R.E.M.", "John Lennon", "Bon Jovi", "Neil Young", "Boston",
        "Def Leppard", "Scorpions", "Heart", "Billy Joel", "Kiss", "Jethro Tull", "Santana", "Steve Miller Band", "Yes",
        "Janis Joplin", "Talking Heads", "Foreigner", "Buddy Holly", "Chicago", "The Velvet Underground",
        "Alice Cooper", "Soundgarden", "Grateful Dead", "Steely Dan", "The Smashing Pumpkins", "Styx",
        "The Cure", "Blondie", "Alice in Chains", "Bad Company", "The Band", "The Yardbirds", "Jefferson Airplane",
        "Oasis", "The Cars", "George Harrison", "Sex Pistols", "Supertramp", "The Moody Blues", "Cheap Trick",
        "Linkin Park", "Rage Against the Machine", "The Jimi Hendrix Experience", "Greta Van Fleet",
        "Panic! At The Disco", "PVRIS", "​twenty one pilots", "Mumford & Sons", "Fall Out Boy",

    ]
}

REGGAE_ARTISTS = {
    "genre": "Reggae",
    "artists": [
        "Bob Marley & The Wailers", "Jimmy Cliff", "Peter Tosh", "Burning Spear", "Ruth Brown",
        "Bunny Wailer", "Gregory Isaacs", "Ziggy Marley", "UB40", "Buju Banton", "Black Uhuru", "Desmond Dekker",
        "Beres Hammond", "Chronixx", "Shaggy", "Capleton", "Max Romeo", "Rita Marley",
        "Freddie McGregor", "Damian Marley", "Israel Vibration", "Steel Pulse", "Beenie Man", "Jah Cure", "Gentleman",
        "Stephen Marley", "Maxi Priest", "Luciano", "The Abyssinians", "Sizzla", "Bounty Killer",
        "Lucky Dube", "Alton Ellis", "John Holt", "Protoje", "The Upsetters", "Tarrus Riley"
    ]
}

ARTISTS_BY_GENRE = {
    "Pop": POP_ARTISTS,
    "Rap": RAP_ARTISTS,
    "Country": COUNTRY_ARTISTS,
    "Rock": ROCK_ARTISTS,
    "Reggae": REGGAE_ARTISTS
}

save_dir = U.make_dir(os.path.dirname(__file__) + "/datasets")
save_path = os.path.join(save_dir, "genius_lyrics_v2.csv")


for key in ARTISTS_BY_GENRE.keys():
    for artist in ARTISTS_BY_GENRE[key]["artists"]:
        data = []
        artist_obj = genius.search_artist(artist, max_songs=60, sort="popularity")

        if artist_obj is not None:
            for i in range(artist_obj.num_songs):
                song = artist_obj.songs[i]

                if None not in (song.artist, song.year, song.album, song.title, song.lyrics):
                    data.append((ARTISTS_BY_GENRE[key]["genre"], song.artist, song.year[:4], song.album, song.title, song.lyrics))

            # Save after each artist for safety
            U.save_dataset(save_path, data, append=True)


### Now we will perform a scraping check
all_songs = U.load_dataset(save_path)
print("\n\nThe total amount of scraped songs: {}".format(len(all_songs)))

all_artists = []
for key in ARTISTS_BY_GENRE.keys():
    all_artists += ARTISTS_BY_GENRE[key]["artists"]

specified_duplicate_artists = set([x for x in all_artists if all_artists.count(x) > 1])
print("The following {} artists were duplicate: {}".format(len(specified_duplicate_artists), specified_duplicate_artists))

unique_found_songs = set([lyrics[5] for lyrics in all_songs])
print("There are duplicate songs {} found.".format(len(all_songs)-len(unique_found_songs)))

all_artists = set(all_artists)
unique_found_artists = set([lis[1] for lis in all_songs])
not_found_artists = all_artists.difference(unique_found_artists)
print("The scraped data does not contain songs from the following {} artists:\n{}".format(len(not_found_artists),
                                                                                                 not_found_artists))
